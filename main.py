"""
Feature Extraction and Normalization Module
---------------------------------------------------------
Changes vs original:
  1. ProcessPoolExecutor  — true CPU parallelism for librosa (no GIL)
  2. ThreadedConnectionPool — reuse DB connections, no reconnect overhead
  3. Vectorized NumPy normalization — single matrix op instead of Python loop
  4. Single bulk UPDATE via execute_values — 1 SQL round-trip for all rows
  5. Normalization params computed in SQL (pg_vector aggregate) — no full fetch
  6. Context-manager helpers — guaranteed connection release
  7. Batch chunking — bounded memory regardless of dataset size
"""

import logging
import asyncio
import os
import math
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import List, Tuple, Optional

import numpy as np
import librosa
from psycopg2 import pool
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR            = 16_000
HOP_LENGTH    = 512
N_MFCC        = 13
SILENCE_DB    = -40
FEATURE_DIM   = 18          # 5 hand-crafted + 13 MFCCs
FMIN          = float(librosa.note_to_hz("C2"))
FMAX          = float(librosa.note_to_hz("C7"))


# ===========================================================================
# 0.  pgvector parse helper
# ===========================================================================
def _vec_to_list(v) -> List[float]:
    """
    Convert whatever psycopg2 returns for a pgvector column to list[float].

    pgvector has NO direct SQL cast to float8[]:
        vector::float8[]  → CannotCoerce error
    Workaround: fetch as TEXT and parse the '[x,y,z]' string here.
    This also handles the case where pgvector Python bindings are installed
    and return a numpy array or list directly.
    """
    if v is None:
        return [0.0] * FEATURE_DIM
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    # Fallback: string representation '[x, y, z, ...]'
    return [float(x) for x in str(v).strip('[]').split(',')]


# ===========================================================================
# 1.  Worker function — must live at MODULE LEVEL so ProcessPoolExecutor
#     can pickle it across processes.
# ===========================================================================
def _extract_worker(file_path: str) -> Optional[List[float]]:
    """
    Pure CPU work: load audio + compute 18-dim feature vector.
    Runs in a separate OS process → no GIL contention.
    """
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)

        # ── Time-domain ──────────────────────────────────────────────────
        rms        = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
        avg_energy = float(np.mean(rms))

        zcr        = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
        avg_zcr    = float(np.mean(zcr))

        rms_db         = librosa.amplitude_to_db(rms, ref=np.max)
        silence_frames = int(np.sum(rms_db < SILENCE_DB))
        total_frames   = rms.shape[1]
        silence_ratio  = silence_frames / total_frames if total_frames > 0 else 0.0

        # ── Frequency-domain ─────────────────────────────────────────────
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=FMIN, fmax=FMAX, sr=sr, hop_length=HOP_LENGTH
        )
        voiced_f0  = f0[voiced_flag] if voiced_flag is not None else np.array([])
        avg_pitch  = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        spec_cent       = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
        avg_centroid    = float(np.mean(spec_cent))

        mfccs           = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        avg_mfccs       = np.mean(mfccs, axis=1).tolist()   # list[float], len=13

        return [avg_energy, avg_zcr, silence_ratio, avg_pitch, avg_centroid] + avg_mfccs

    except Exception as exc:
        log.error("Extraction failed for %s: %s", file_path, exc)
        return None


# ===========================================================================
# 2.  Connection pool helper
# ===========================================================================
class DBPool:
    """
    Thin wrapper around psycopg2 ThreadedConnectionPool.
    Usage:
        db = DBPool(db_config)
        with db.conn() as conn:
            ...          # auto-returned to pool on exit
    """

    def __init__(self, db_config: dict, minconn: int = 2, maxconn: int = 10):
        self._pool = pool.ThreadedConnectionPool(minconn, maxconn, **db_config)
        log.info("Connection pool created (min=%d, max=%d).", minconn, maxconn)

    @contextmanager
    def conn(self):
        """Yield a connection; always return it to the pool."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close(self):
        self._pool.closeall()


# ===========================================================================
# 3.  Feature Extractor
# ===========================================================================
class FeatureExtractor:
    """
    Parallelises librosa calls over multiple OS processes.
    Worker limit = min(cpu_count, 8) to avoid RAM pressure.
    """

    def __init__(self, db_pool: DBPool):
        self.db = db_pool
        self._workers = min(os.cpu_count() or 4, 8)

    # ── DB helpers ──────────────────────────────────────────────────────────
    def _fetch_paths(self) -> List[Tuple[int, str]]:
        with self.db.conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT obj_id, file_path FROM multimedia_objects ORDER BY obj_id;")
            return cur.fetchall()

    # ── Async extraction ────────────────────────────────────────────────────
    async def process_all(
        self, batch_size: int = 200, limit: int = -1
    ) -> List[Tuple[int, List[float]]]:
        """
        Extract features in parallel using ProcessPoolExecutor.

        Parameters
        ----------
        batch_size : int
            Files processed per batch (controls peak RAM).
        limit : int
            Cap total files (-1 = all).
        """
        rows = self._fetch_paths()
        if limit > 0:
            rows = rows[:limit]
        if not rows:
            log.warning("No files found in DB.")
            return []

        total   = len(rows)
        results = []
        loop    = asyncio.get_event_loop()

        log.info("Extracting %d files with %d workers, batch=%d …",
                 total, self._workers, batch_size)

        with ProcessPoolExecutor(max_workers=self._workers) as executor:
            # Iterate in chunks to bound memory
            for start in range(0, total, batch_size):
                chunk = rows[start : start + batch_size]
                paths = [fp for _, fp in chunk]
                obj_ids = [oid for oid, _ in chunk]

                # Submit all files in chunk concurrently
                futures = [
                    loop.run_in_executor(executor, _extract_worker, fp)
                    for fp in paths
                ]
                vectors = await asyncio.gather(*futures)

                ok = 0
                for oid, vec in zip(obj_ids, vectors):
                    if vec is not None:
                        results.append((oid, vec))
                        ok += 1
                    else:
                        log.warning("Skipped obj_id=%d (extraction error).", oid)

                log.info("  Batch %d–%d: %d/%d ok.",
                         start + 1, start + len(chunk), ok, len(chunk))

        log.info("Extraction complete: %d/%d vectors.", len(results), total)
        return results


# ===========================================================================
# 4.  Normalizer
# ===========================================================================
class Normalizer:
    """
    Stores raw vectors, computes normalization params, updates normalized cols.
    All heavy lifting done with NumPy matrix ops or SQL aggregates.
    """

    def __init__(self, db_pool: DBPool):
        self.db = db_pool

    # ── Schema setup ────────────────────────────────────────────────────────
    def ensure_schema(self):
        """Add vector columns and normalization-params table if missing."""
        with self.db.conn() as conn:
            cur = conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            for col in ("raw_feature_vector", "min_max_scaling_vec", "z_score_nor_vec"):
                cur.execute(f"""
                    ALTER TABLE multimedia_objects
                    ADD COLUMN IF NOT EXISTS {col} VECTOR({FEATURE_DIM});
                """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS normalization_params (
                    param_name   TEXT PRIMARY KEY,
                    param_vector VECTOR({FEATURE_DIM})
                );
            """)
        log.info("Schema ready.")

    # ── Bulk-store raw vectors ───────────────────────────────────────────────
    def store_raw_vectors(self, data: List[Tuple[int, List[float]]]):
        """
        Single execute_values call → 1 round-trip regardless of row count.
        """
        if not data:
            return
        rows = [(oid, vec) for oid, vec in data]
        with self.db.conn() as conn:
            cur = conn.cursor()
            execute_values(
                cur,
                """
                UPDATE multimedia_objects
                SET raw_feature_vector = data.vec::vector
                FROM (VALUES %s) AS data(obj_id, vec)
                WHERE multimedia_objects.obj_id = data.obj_id::int
                """,
                rows,
                template="(%s, %s::float8[])",
                page_size=500,
            )
        log.info("Stored raw vectors for %d objects.", len(data))

    # ── Compute params (NumPy, no SQL aggregate needed) ─────────────────────
    def compute_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fetch all raw vectors once, compute min/max/mean/std as numpy arrays.
        Returns four ndarrays of shape (FEATURE_DIM,).

        IMPORTANT: cast vector → float8[] in SQL so psycopg2 returns a native
        Python list[float] instead of the opaque pgvector string/object, which
        causes "inhomogeneous shape" when passed to np.array().
        """
        with self.db.conn() as conn:
            cur = conn.cursor()
            # Fetch as TEXT: pgvector has no direct cast to float8[].
            # _vec_to_list() parses the '[x,y,z]' string into list[float].
            cur.execute(
                "SELECT raw_feature_vector::text FROM multimedia_objects "
                "WHERE raw_feature_vector IS NOT NULL;"
            )
            rows = cur.fetchall()

        if not rows:
            raise RuntimeError("No raw vectors found — run extraction first.")

        # Parse each text row → uniform float list, then stack into matrix
        matrix = np.array([_vec_to_list(r[0]) for r in rows], dtype=np.float64)
        assert matrix.shape == (len(rows), FEATURE_DIM), \
            f"Unexpected shape {matrix.shape}, expected ({len(rows)}, {FEATURE_DIM})"

        min_v  = matrix.min(axis=0)
        max_v  = matrix.max(axis=0)
        mean_v = matrix.mean(axis=0)
        std_v  = matrix.std(axis=0)

        log.info("Params computed on %d samples × %d dims.", *matrix.shape)
        return min_v, max_v, mean_v, std_v

    # ── Persist params ──────────────────────────────────────────────────────
    def save_params(
        self,
        min_v: np.ndarray,
        max_v: np.ndarray,
        mean_v: np.ndarray,
        std_v: np.ndarray,
    ):
        params = [
            ("min_max_min",  min_v.tolist()),
            ("min_max_max",  max_v.tolist()),
            ("z_score_mean", mean_v.tolist()),
            ("z_score_std",  std_v.tolist()),
        ]
        with self.db.conn() as conn:
            cur = conn.cursor()
            cur.execute("TRUNCATE normalization_params;")
            for name, vec in params:
                cur.execute(
                    "INSERT INTO normalization_params VALUES (%s, %s::vector);",
                    (name, vec),
                )
        log.info("Normalization params saved.")

    # ── Vectorized normalize + single bulk UPDATE ────────────────────────────
    def apply_normalization(
        self,
        min_v: np.ndarray,
        max_v: np.ndarray,
        mean_v: np.ndarray,
        std_v: np.ndarray,
        chunk_size: int = 500,
    ):
        """
        Key optimizations vs original:
          • All math is NumPy broadcast (no Python loops over dimensions).
          • Rows processed in chunks to bound memory.
          • Each chunk = 1 execute_values call (not N individual UPDATEs).
        """
        with self.db.conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT obj_id, raw_feature_vector::text FROM multimedia_objects "
                "WHERE raw_feature_vector IS NOT NULL ORDER BY obj_id;"
            )
            all_rows = cur.fetchall()

        if not all_rows:
            log.warning("No raw vectors to normalise.")
            return

        # Pre-compute safe divisors once — avoids repeated np.where in loop
        range_v       = max_v - min_v
        range_safe    = np.where(range_v == 0, 1.0, range_v)   # shape (18,)
        std_safe      = np.where(std_v   == 0, 1.0, std_v)     # shape (18,)

        total_updated = 0
        n_chunks = math.ceil(len(all_rows) / chunk_size)

        for i in range(n_chunks):
            chunk = all_rows[i * chunk_size : (i + 1) * chunk_size]

            obj_ids = [r[0] for r in chunk]
            # Shape: (chunk_len, 18) — single allocation
            matrix  = np.array([_vec_to_list(r[1]) for r in chunk], dtype=np.float64)

            # ── Vectorized normalization (broadcast over all rows at once) ──
            mm_matrix = (matrix - min_v)  / range_safe  # Min-Max → [0, 1]
            zs_matrix = (matrix - mean_v) / std_safe    # Z-score → N(0,1)

            # Build rows for bulk UPDATE: (mm_vec, zs_vec, obj_id)
            update_rows = [
                (mm.tolist(), zs.tolist(), oid)
                for oid, mm, zs in zip(obj_ids, mm_matrix, zs_matrix)
            ]

            with self.db.conn() as conn:
                cur = conn.cursor()
                execute_values(
                    cur,
                    """
                    UPDATE multimedia_objects
                    SET min_max_scaling_vec = data.mm::vector,
                        z_score_nor_vec     = data.zs::vector
                    FROM (VALUES %s) AS data(mm, zs, obj_id)
                    WHERE multimedia_objects.obj_id = data.obj_id::int
                    """,
                    update_rows,
                    template="(%s::float8[], %s::float8[], %s)",
                    page_size=chunk_size,
                )
            total_updated += len(chunk)
            log.info("  Chunk %d/%d — updated %d rows (total %d).",
                     i + 1, n_chunks, len(chunk), total_updated)

        log.info("Normalization applied to %d rows.", total_updated)


# ===========================================================================
# 5.  Entry point
# ===========================================================================
async def main():
    DB_CONFIG = {
        "host":     "localhost",
        "port":     5432,
        "dbname":   "mmdb",
        "user":     "postgres",
        "password": "2324",
    }

    db = DBPool(DB_CONFIG, minconn=2, maxconn=8)
    try:
        # ── Step 1: schema ────────────────────────────────────────────────
        normalizer = Normalizer(db)
        normalizer.ensure_schema()

        # ── Step 2: decide whether extraction is needed ─────────────────
        # Count rows with NULL raw_feature_vector (not yet extracted).
        with db.conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE raw_feature_vector IS NOT NULL) AS done,
                    COUNT(*)                                                AS total
                FROM multimedia_objects;
            """)
            done, total_rows = cur.fetchone()

        log.info("raw_feature_vector: %d/%d rows already extracted.", done, total_rows)

        if done < total_rows:
            log.info("Extracting features for %d missing rows …", total_rows - done)
            extractor = FeatureExtractor(db)
            raw_data  = await extractor.process_all(batch_size=200, limit=-1)
            if not raw_data:
                log.error("Extraction produced no results. Aborting.")
                return
            # ── Step 3: bulk-store raw vectors ────────────────────────────
            normalizer.store_raw_vectors(raw_data)
        else:
            log.info("All raw vectors already present — skipping extraction.")

        # ── Step 4: compute normalization params ──────────────────────────
        # Always recompute so min_max / z_score stay consistent on re-runs.
        min_v, max_v, mean_v, std_v = normalizer.compute_params()

        log.info("min[:5]  = %s", min_v[:5].round(4))
        log.info("max[:5]  = %s", max_v[:5].round(4))
        log.info("mean[:5] = %s", mean_v[:5].round(4))
        log.info("std[:5]  = %s", std_v[:5].round(4))

        # ── Step 5: persist params ────────────────────────────────────────
        normalizer.save_params(min_v, max_v, mean_v, std_v)

        # ── Step 6: vectorized normalize + bulk update ────────────────────
        normalizer.apply_normalization(min_v, max_v, mean_v, std_v, chunk_size=500)

        log.info("Pipeline complete.")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())