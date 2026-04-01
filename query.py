"""
query.py — Voice similarity search: 6 combinations + Precision/Recall
----------------------------------------------------------------------
Combinations
  #  | Normalization | Distance  | Index
  ───|───────────────|───────────|──────────────
  1  | min-max       | euclidean | ivfflat (<->)
  2  | min-max       | cosine    | ivfflat (<=>)
  3  | z-score       | euclidean | ivfflat (<->)
  4  | z-score       | cosine    | ivfflat (<=>)
  5  | min-max       | euclidean | cKDTree
  6  | z-score       | euclidean | cKDTree

  cKDTree + cosine is intentionally excluded: scipy cKDTree only supports
  Minkowski metrics (p-norms); cosine similarity requires a different structure.

Precision & Recall are computed per-query using speaker_name as ground truth:
  Precision@k = |retrieved ∩ relevant| / k
  Recall@k    = |retrieved ∩ relevant| / |relevant in DB|

  "relevant" = all DB rows sharing the same speaker_name as the query file,
               excluding the query file itself if it is already in the DB.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np

from main import _extract_worker, _vec_to_list, DBPool, DB_CONFIG
from k_d_tree import sim as kdtree_sim, build_trees

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Result type: (obj_id, speaker_name, score)
SearchResult = List[Tuple[int, str, float]]


# ===========================================================================
# 1.  Normalization helpers
# ===========================================================================
def _load_norm_params(db: DBPool) -> Dict[str, np.ndarray]:
    """
    Fetch all 4 rows from normalization_params in one query.
    Returns dict: {'min_max_min': ndarray, 'min_max_max': ...,
                   'z_score_mean': ..., 'z_score_std': ...}

    IMPORTANT: cur.execute() is INSIDE 'with db.conn()' block.
    """
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT param_name, param_vector::text FROM normalization_params;")
        rows = cur.fetchall()

    if len(rows) < 4:
        raise RuntimeError(
            f"normalization_params has {len(rows)} rows — expected 4. "
            "Run main.py first."
        )
    return {name: np.array(_vec_to_list(vec), dtype=np.float64)
            for name, vec in rows}


def normalize_query(
    raw_vec: List[float],
    params: Dict[str, np.ndarray],
) -> Tuple[List[float], List[float]]:
    """
    Apply both normalizations to a raw 18-dim feature vector.

    Parameters
    ----------
    raw_vec : raw feature vector from _extract_worker()
    params  : dict from _load_norm_params()

    Returns
    -------
    (minmax_vec, zscore_vec) — both as list[float]
    """
    v         = np.array(raw_vec, dtype=np.float64)
    min_v     = params["min_max_min"]
    max_v     = params["min_max_max"]
    mean_v    = params["z_score_mean"]
    std_v     = params["z_score_std"]

    range_safe = np.where((max_v - min_v) == 0, 1.0, max_v - min_v)
    std_safe   = np.where(std_v == 0, 1.0, std_v)

    mm = (v - min_v) / range_safe
    zs = (v - mean_v) / std_safe

    return mm.tolist(), zs.tolist()


# ===========================================================================
# 2.  ivfflat search (pgvector)
# ===========================================================================
def search_ivfflat(
    query_vec: List[float],
    norm: str,        # 'minmax' | 'zscore'
    metric: str,      # 'euclidean' | 'cosine'
    db: DBPool,
    k: int = 10,
    probes: int = 6,
) -> SearchResult:
    """
    ANN search via pgvector ivfflat index.

    Operators:
      euclidean → <->   (vector_l2_ops)
      cosine    → <=>   (vector_cosine_ops)

    Parameters
    ----------
    norm   : which normalised column to query
    metric : distance metric
    probes : ivfflat.probes — higher = more accurate, slower
    """
    col_map = {
        "minmax": "min_max_scaling_vec",
        "zscore": "z_score_nor_vec",
    }
    op_map = {
        "euclidean": "<->",
        "cosine":    "<=>",
    }
    if norm not in col_map:
        raise ValueError(f"norm must be 'minmax' or 'zscore', got {norm!r}")
    if metric not in op_map:
        raise ValueError(f"metric must be 'euclidean' or 'cosine', got {metric!r}")

    col      = col_map[norm]
    operator = op_map[metric]
    vec_str  = "[" + ",".join(map(str, query_vec)) + "]"

    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute(f"SET ivfflat.probes = {probes};")
        cur.execute(f"""
            SELECT obj_id,
                   speaker_name,
                   {col} {operator} %s::vector AS distance
            FROM   multimedia_objects
            ORDER  BY distance
            LIMIT  %s;
        """, (vec_str, k))
        rows = cur.fetchall()

    return [(r[0], r[1] or "", float(r[2])) for r in rows]


# ===========================================================================
# 3.  cKDTree search (euclidean only)
# ===========================================================================
def search_kdtree(
    query_vec: List[float],
    norm: str = "minmax",
    k: int = 10,
) -> SearchResult:
    """
    Exact nearest-neighbour search via scipy cKDTree (euclidean).

    Parameters
    ----------
    norm : 'minmax' | 'zscore' — selects which pre-built tree to use
    """
    return kdtree_sim(query_vec, norm=norm, k=k)


# ===========================================================================
# 4.  Precision & Recall
# ===========================================================================
def _get_relevant_ids(
    query_speaker: str,
    query_file_path: Optional[str],
    db: DBPool,
) -> List[int]:
    """
    Return all obj_ids in DB that share speaker_name with the query,
    excluding the query file itself (if it is already in the DB).

    NOTE: cur operations are INSIDE 'with db.conn()' block.
    """
    with db.conn() as conn:
        cur = conn.cursor()
        # if query_file_path:
        #     cur.execute("""
        #         SELECT obj_id FROM multimedia_objects
        #         WHERE  speaker_name = %s
        #           AND  file_path   != %s;
        #     """, (query_speaker, query_file_path))
        # else:
        #     cur.execute("""
        #         SELECT obj_id FROM multimedia_objects
        #         WHERE  speaker_name = %s;
        #     """, (query_speaker,))
        cur.execute("""
            SELECT obj_id FROM multimedia_objects
            WHERE  speaker_name = %s;
        """, (query_speaker,))
        rows = cur.fetchall()

    return [r[0] for r in rows]


def precision_recall(
    results: SearchResult,
    query_speaker: str,
    query_file_path: Optional[str],
    db: DBPool,
) -> Tuple[float, float]:
    """
    Compute Precision@k and Recall@k.

    Ground truth = all DB rows with the same speaker_name as the query file,
                   minus the query file itself.

    Precision@k = |retrieved ∩ relevant| / k
    Recall@k    = |retrieved ∩ relevant| / |relevant|

    Parameters
    ----------
    results          : output of search_ivfflat() or search_kdtree()
    query_speaker    : speaker_name of the query file
    query_file_path  : absolute path of query file (to exclude from relevant set)
    db               : DBPool

    Returns
    -------
    (precision, recall) as floats in [0, 1]
    """
    if not query_speaker:
        log.warning("query_speaker is empty — cannot compute precision/recall.")
        return 0.0, 0.0

    relevant = set(_get_relevant_ids(query_speaker, query_file_path, db))
    if not relevant:
        log.warning("No other files for speaker '%s' in DB.", query_speaker)
        return 0.0, 0.0

    retrieved = {r[0] for r in results}
    hits      = len(retrieved & relevant)
    k         = len(results)

    precision = hits / k         if k         > 0 else 0.0
    recall    = hits / len(relevant)

    return precision, recall


# ===========================================================================
# 5.  Pretty printer
# ===========================================================================
def _print_results(
    label: str,
    results: SearchResult,
    precision: float,
    recall: float,
) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print(f"{'─' * width}")
    print(f"  {'#':<4} {'obj_id':<8} {'speaker':<20} {'distance':>10}")
    print(f"  {'─'*4} {'─'*8} {'─'*20} {'─'*10}")
    for rank, (obj_id, speaker, dist) in enumerate(results, 1):
        print(f"  {rank:<4} {obj_id:<8} {speaker:<20} {dist:>10.6f}")
    print(f"\n  Precision@{len(results)}: {precision:.4f}   Recall@{len(results)}: {recall:.4f}")
    print(f"{'─' * width}")


# ===========================================================================
# 6.  Main — run all 6 combinations
# ===========================================================================
def run_all(
    file_path: str,
    query_speaker: Optional[str] = None,
    k: int = 10,
) -> None:
    """
    Extract features from file_path, run all 6 search combinations,
    print ranked results + Precision@k / Recall@k for each.

    Parameters
    ----------
    file_path      : absolute path to the query .flac/.wav/.mp3 file
    query_speaker  : known speaker name (used for P/R); auto-detected from DB
                     if the file is already in the corpus.
    k              : number of results to return per search
    """
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return

    db = DBPool(DB_CONFIG, minconn=1, maxconn=6)
    try:
        # ── Step 1: extract raw features ─────────────────────────────────
        log.info("Extracting features from %s …", path.name)
        raw_vec = _extract_worker(str(path))
        if raw_vec is None:
            print("[ERROR] Feature extraction failed.")
            return

        # ── Step 2: load norm params and normalise ────────────────────────
        params = _load_norm_params(db)
        mm_vec, zs_vec = normalize_query(raw_vec, params)

        # ── Step 3: auto-detect speaker if file is in DB ──────────────────
        detected_speaker: Optional[str] = query_speaker
        if detected_speaker is None:
            with db.conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT speaker_name FROM multimedia_objects WHERE file_path = %s;",
                    (str(path),),
                )
                row = cur.fetchone()
            if row:
                detected_speaker = row[0] or ""
                log.info("Speaker auto-detected from DB: '%s'", detected_speaker)
            else:
                log.warning("File not in DB — speaker unknown; P/R will be 0.")
                detected_speaker = ""

        # ── Step 4: ensure KD-trees exist ────────────────────────────────
        build_trees(db=db, force=False)

        # ── Step 5: run all 6 combinations ───────────────────────────────
        combos = [
            # (label,                          search_fn_args,           vec)
            ("ivfflat | min-max  | euclidean",
             dict(norm="minmax", metric="euclidean", db=db, k=k),      mm_vec),
            ("ivfflat | min-max  | cosine   ",
             dict(norm="minmax", metric="cosine",    db=db, k=k),      mm_vec),
            ("ivfflat | z-score  | euclidean",
             dict(norm="zscore", metric="euclidean", db=db, k=k),      zs_vec),
            ("ivfflat | z-score  | cosine   ",
             dict(norm="zscore", metric="cosine",    db=db, k=k),      zs_vec),
            ("kdtree  | min-max  | euclidean",
             dict(norm="minmax", k=k),                                  mm_vec),
            ("kdtree  | z-score  | euclidean",
             dict(norm="zscore", k=k),                                  zs_vec),
        ]

        print(f"\n{'='*60}")
        print(f"  Query file : {path.name}")
        print(f"  Speaker    : {detected_speaker or '(unknown)'}")
        print(f"  Top-k      : {k}")
        print(f"{'='*60}")

        for label, kwargs, vec in combos:
            use_kdtree = label.startswith("kdtree")
            if use_kdtree:
                results = search_kdtree(vec, **kwargs)
            else:
                results = search_ivfflat(vec, **kwargs)
            # print(len(results))
            prec, rec = precision_recall(
                results,
                query_speaker=detected_speaker,
                query_file_path=str(path),
                db=db,
            )
            _print_results(label, results, prec, rec)

    finally:
        db.close()


# ===========================================================================
# 7.  Entry point
# ===========================================================================
if __name__ == "__main__":
    # Example: query with a file already in the corpus (speaker auto-detected)
    QUERY_FILE = "/home/a/static/LibriSpeech/dev-clean/652_129742_652-129742-0000.flac"

    # Optionally override speaker name for files NOT in the corpus:
    # run_all(QUERY_FILE, query_speaker="251", k=10)
    run_all(QUERY_FILE, k=5)
