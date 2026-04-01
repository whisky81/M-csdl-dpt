"""
k_d_mean.py — KD-tree builder and similarity search
----------------------------------------------------
Builds two separate cKDTrees (one for min-max vectors, one for z-score vectors),
persists them as pickles, and exposes a sim() function used by query.py.

Each tree maps array index → (obj_id, speaker_name) so query.py can compute
precision/recall without an extra DB round-trip.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from main import DBPool, _vec_to_list, DB_CONFIG

log = logging.getLogger(__name__)

# Pickle file paths
_PICKLE_MINMAX = Path("kdtree_minmax.pkl")
_PICKLE_ZSCORE = Path("kdtree_zscore.pkl")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_vectors(db: DBPool) -> Tuple[
    List[Tuple[int, str]],  # mapping: [(obj_id, speaker_name), ...]
    List[List[float]],      # min-max vectors
    List[List[float]],      # z-score vectors
]:
    """
    Fetch obj_id, speaker_name, min_max_scaling_vec, z_score_nor_vec
    from DB in one query.  Returns parallel lists.

    NOTE: ALL cursor operations are inside the 'with db.conn()' block.
    """
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT obj_id,
                   speaker_name,
                   min_max_scaling_vec::text,
                   z_score_nor_vec::text
            FROM   multimedia_objects
            WHERE  min_max_scaling_vec IS NOT NULL
              AND  z_score_nor_vec     IS NOT NULL
            ORDER BY obj_id;
        """)
        rows = cur.fetchall()

    mapping   = [(r[0], r[1] or "") for r in rows]
    mm_vecs   = [_vec_to_list(r[2]) for r in rows]
    zs_vecs   = [_vec_to_list(r[3]) for r in rows]
    return mapping, mm_vecs, zs_vecs


def _build_one(
    points: List[List[float]],
    mapping: List[Tuple[int, str]],
    pickle_path: Path,
    label: str,
) -> Tuple[cKDTree, List[Tuple[int, str]]]:
    """Build cKDTree from points and persist to pickle."""
    arr  = np.array(points, dtype=np.float64)
    tree = cKDTree(arr, leafsize=16)
    with open(pickle_path, "wb") as f:
        pickle.dump((tree, mapping), f)
    log.info("Built & saved %s KD-tree (%d points) → %s", label, len(points), pickle_path)
    return tree, mapping


def _load_one(
    pickle_path: Path,
    label: str,
) -> Optional[Tuple[cKDTree, List[Tuple[int, str]]]]:
    """Load cKDTree from pickle; return None if file missing."""
    if not pickle_path.exists():
        return None
    with open(pickle_path, "rb") as f:
        tree, mapping = pickle.load(f)
    log.info("Loaded %s KD-tree (%d points) from %s", label, tree.n, pickle_path)
    return tree, mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_trees(db: Optional[DBPool] = None, force: bool = False) -> None:
    """
    Build (or rebuild) both KD-trees and save to disk.

    Parameters
    ----------
    db    : DBPool — pass an existing pool; a new one is created if None.
    force : bool  — rebuild even if pickle files already exist.
    """
    both_exist = _PICKLE_MINMAX.exists() and _PICKLE_ZSCORE.exists()
    if both_exist and not force:
        log.info("KD-tree pickles already exist. Use force=True to rebuild.")
        return

    close_after = False
    if db is None:
        db = DBPool(DB_CONFIG, minconn=1, maxconn=4)
        close_after = True

    try:
        mapping, mm_vecs, zs_vecs = _load_vectors(db)
        if not mapping:
            log.error("No normalized vectors in DB — run main.py first.")
            return
        _build_one(mm_vecs, mapping, _PICKLE_MINMAX, "min-max")
        _build_one(zs_vecs, mapping, _PICKLE_ZSCORE, "z-score")
    finally:
        if close_after:
            db.close()


def _get_tree(norm: str) -> Tuple[cKDTree, List[Tuple[int, str]]]:
    """
    Load a cKDTree from pickle (build it first if missing).

    Parameters
    ----------
    norm : 'minmax' | 'zscore'
    """
    if norm == "minmax":
        result = _load_one(_PICKLE_MINMAX, "min-max")
        if result is None:
            log.info("min-max pickle not found — building now …")
            build_trees()
            result = _load_one(_PICKLE_MINMAX, "min-max")
    elif norm == "zscore":
        result = _load_one(_PICKLE_ZSCORE, "z-score")
        if result is None:
            log.info("z-score pickle not found — building now …")
            build_trees()
            result = _load_one(_PICKLE_ZSCORE, "z-score")
    else:
        raise ValueError(f"norm must be 'minmax' or 'zscore', got {norm!r}")

    if result is None:
        raise RuntimeError(f"Could not load or build KD-tree for norm={norm!r}")
    return result


def sim(
    query_vec: List[float],
    norm: str = "minmax",
    k: int = 5,
) -> List[Tuple[int, str, float]]:
    """
    Find k nearest neighbours using the cKDTree (Euclidean distance).

    Parameters
    ----------
    query_vec : 18-dim list[float] — already normalised (min-max or z-score).
    norm      : 'minmax' | 'zscore' — selects which tree to query.
    k         : number of results.

    Returns
    -------
    List of (obj_id, speaker_name, distance) sorted ascending by distance.
    """
    tree, mapping = _get_tree(norm)
    q = np.array(query_vec, dtype=np.float64)
    distances, indices = tree.query(q, k=k)

    # tree.query returns scalars when k=1, ensure iterables
    if k == 1:
        distances = [distances]
        indices   = [indices]

    results = []
    for idx, dist in zip(indices, distances):
        obj_id, speaker_name = mapping[idx]
        results.append((obj_id, speaker_name, float(dist)))
    return results


# ---------------------------------------------------------------------------
# Standalone build (run directly to pre-build trees)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    db = DBPool(DB_CONFIG, minconn=1, maxconn=4)
    try:
        build_trees(db=db, force=True)
    finally:
        db.close()
