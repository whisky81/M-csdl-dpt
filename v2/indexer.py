# indexer.py (parallel version)
import os
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import asyncio

import numpy as np
from scipy.spatial import cKDTree
from psycopg2.extras import execute_values

from db import DBPool, parse_vector_str
from extractor import process_file  # hàm này xử lý 1 file và trả về list segment info
from normalizer import compute_global_minmax, save_minmax_params, minmax_normalize
from config import FEATURE_DIM

log = logging.getLogger(__name__)

KD_TREE_PICKLE = Path("kdtree_segments_minmax.pkl")

# ----------------------------------------------------------------------
# Worker function (module-level để pickle được)
# ----------------------------------------------------------------------
def _process_file_worker(file_path: str) -> Tuple[str, List[Tuple[int, float, float, List[float]]]]:
    """
    Xử lý một file: trả về (file_path, list of (segment_index, start, end, raw_vec))
    Lưu ý: không truyền file_id vì worker không biết, ta sẽ map sau.
    """
    segments_info = process_file(file_path)
    if not segments_info:
        return (file_path, [])
    result = []
    for idx, (_, start, end, raw_vec) in enumerate(segments_info):
        result.append((idx, start, end, raw_vec))
    return (file_path, result)


# ----------------------------------------------------------------------
# Schema & indexing functions
# ----------------------------------------------------------------------
def build_schema(db: DBPool):
    """Tạo bảng voices_metadata và segments (nếu chưa có)."""
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS voices_metadata (
                file_id          SERIAL PRIMARY KEY,
                file_name        TEXT NOT NULL,
                file_path        TEXT NOT NULL UNIQUE,
                speaker_name     TEXT,
                file_size_bytes  BIGINT,
                word_count       INT,
                duration_seconds FLOAT
            );
        """)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS segments (
                segment_id      SERIAL PRIMARY KEY,
                file_id         INT NOT NULL REFERENCES voices_metadata(file_id) ON DELETE CASCADE,
                segment_index   INT NOT NULL,
                start_time      FLOAT NOT NULL,
                end_time        FLOAT NOT NULL,
                raw_vec         VECTOR({FEATURE_DIM}) NOT NULL,
                minmax_vec      VECTOR({FEATURE_DIM}) NOT NULL,
                UNIQUE (file_id, segment_index)
            );
        """)
    log.info("Schema created/verified.")


async def index_all_files_parallel(db: DBPool, rebuild: bool = False, max_workers: int = None):
    """
    Duyệt tất cả file trong voices_metadata, trích xuất segment song song.
    Sử dụng ProcessPoolExecutor và asyncio để tối ưu.
    """
    # Kiểm tra segments hiện có
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM segments;")
        seg_count = cur.fetchone()[0]
        if seg_count > 0 and not rebuild:
            log.info("Segments already exist (%d rows). Skipping extraction.", seg_count)
            return

        cur.execute("SELECT file_id, file_path FROM voices_metadata ORDER BY file_id;")
        files = cur.fetchall()

    total = len(files)
    log.info("Processing %d files using %d workers...", total, max_workers or (os.cpu_count() or 4))

    # Map file_path -> file_id
    path_to_id = {fp: fid for fid, fp in files}
    all_paths = [fp for _, fp in files]

    # Thực hiện song song
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [loop.run_in_executor(executor, _process_file_worker, fp) for fp in all_paths]
        results = await asyncio.gather(*futures)

    # Gom dữ liệu để insert
    segment_data = []  # (file_id, seg_idx, start, end, raw_vec)
    for file_path, seg_list in results:
        file_id = path_to_id[file_path]
        for seg_idx, start, end, raw_vec in seg_list:
            segment_data.append((
                int(file_id),
                int(seg_idx),
                float(start),
                float(end),
                [float(x) for x in raw_vec]
            ))

    log.info("Extracted %d segments in total.", len(segment_data))

    # Lưu vào DB
    with db.conn() as conn:
        cur = conn.cursor()
        if rebuild:
            cur.execute("TRUNCATE segments RESTART IDENTITY CASCADE;")
        execute_values(
            cur,
            """
            INSERT INTO segments (file_id, segment_index, start_time, end_time, raw_vec, minmax_vec)
            VALUES %s
            ON CONFLICT (file_id, segment_index) DO UPDATE SET
                start_time = EXCLUDED.start_time,
                end_time = EXCLUDED.end_time,
                raw_vec = EXCLUDED.raw_vec;
            """,
            [(fid, sidx, st, et, rv, rv) for fid, sidx, st, et, rv in segment_data],
            template="(%s, %s, %s, %s, %s::float8[], %s::float8[])"
        )
    log.info("Segments stored.")


def normalize_all_segments(db: DBPool):
    """Tính global min/max và cập nhật minmax_vec cho tất cả segments."""
    min_vals, max_vals = compute_global_minmax(db)
    save_minmax_params(db, min_vals, max_vals)

    # Cập nhật minmax_vec
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT segment_id, raw_vec::text FROM segments;")
        rows = cur.fetchall()

    update_data = []
    for seg_id, raw_str in rows:
        raw = parse_vector_str(raw_str)
        norm = minmax_normalize(raw, min_vals, max_vals)
        update_data.append((norm, seg_id))

    with db.conn() as conn:
        cur = conn.cursor()
        execute_values(
            cur,
            """
            UPDATE segments SET minmax_vec = data.norm::vector
            FROM (VALUES %s) AS data(norm, seg_id)
            WHERE segments.segment_id = data.seg_id;
            """,
            update_data,
            template="(%s::float8[], %s)"
        )
    log.info("MinMax vectors updated for all segments.")


def create_ivfflat_indexes(db: DBPool):
    """Tạo hai index IVFFlat cho minmax_vec (L2 và cosine)."""
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM segments;")
        n = cur.fetchone()[0]
        lists = max(10, int(np.sqrt(n)))
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_seg_minmax_l2 ON segments
            USING ivfflat (minmax_vec vector_l2_ops) WITH (lists = {lists});
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_seg_minmax_cosine ON segments
            USING ivfflat (minmax_vec vector_cosine_ops) WITH (lists = {lists});
        """)
    log.info("IVFFlat indexes created with lists=%d", lists)


def build_kdtree(db: DBPool) -> Tuple[cKDTree, List[Tuple[int, int, str]]]:
    """
    Xây dựng cKDTree từ tất cả minmax_vec.
    Trả về tree và mapping: mỗi điểm ánh xạ tới (segment_id, file_id, speaker_name).
    """
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT s.segment_id, s.file_id, v.speaker_name, s.minmax_vec::text
            FROM segments s
            JOIN voices_metadata v ON s.file_id = v.file_id
            ORDER BY s.segment_id;
        """)
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No segments found for KD-tree.")

    points = np.array([parse_vector_str(r[3]) for r in rows], dtype=np.float64)
    tree = cKDTree(points, leafsize=16)
    mapping = [(r[0], r[1], r[2] or "") for r in rows]

    with open(KD_TREE_PICKLE, "wb") as f:
        pickle.dump((tree, mapping), f)
    log.info("KD-tree built and saved to %s with %d points", KD_TREE_PICKLE, len(points))
    return tree, mapping


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
async def run_indexing(rebuild: bool = False, max_workers: int = None):
    """Thực hiện toàn bộ quá trình indexing."""
    import os
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 8)  # giới hạn để tránh quá tải I/O

    db = DBPool()
    try:
        build_schema(db)
        await index_all_files_parallel(db, rebuild=rebuild, max_workers=max_workers)
        normalize_all_segments(db)
        create_ivfflat_indexes(db)
        build_kdtree(db)
        log.info("Indexing completed successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # Chạy asyncio
    asyncio.run(run_indexing(rebuild=False))