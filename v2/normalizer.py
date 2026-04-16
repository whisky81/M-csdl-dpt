# normalizer.py
import logging
import numpy as np
from typing import Dict, List, Tuple

from db import DBPool, parse_vector_str
from config import FEATURE_DIM

log = logging.getLogger(__name__)


def compute_global_minmax(db: DBPool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính min và max trên toàn bộ raw vectors trong bảng segments.
    Trả về hai mảng numpy shape (FEATURE_DIM,).
    """
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT raw_vec::text FROM segments;")
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError("No segments found in DB. Run indexer first.")

    matrix = np.array([parse_vector_str(r[0]) for r in rows], dtype=np.float64)
    min_vals = matrix.min(axis=0)
    max_vals = matrix.max(axis=0)
    log.info("Computed global min/max from %d segments.", len(rows))
    return min_vals, max_vals


def save_minmax_params(db: DBPool, min_vals: np.ndarray, max_vals: np.ndarray):
    """Lưu min/max vào bảng normalization_params."""
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS normalization_params (
                param_name TEXT PRIMARY KEY,
                param_vector VECTOR(%s)
            );
        """, (FEATURE_DIM,))
        cur.execute("DELETE FROM normalization_params WHERE param_name IN ('minmax_min', 'minmax_max');")
        cur.execute(
            "INSERT INTO normalization_params VALUES ('minmax_min', %s::vector);",
            (min_vals.tolist(),)
        )
        cur.execute(
            "INSERT INTO normalization_params VALUES ('minmax_max', %s::vector);",
            (max_vals.tolist(),)
        )
    log.info("Saved min/max normalization params.")


def load_minmax_params(db: DBPool) -> Dict[str, np.ndarray]:
    """Tải min/max từ DB."""
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT param_name, param_vector::text FROM normalization_params WHERE param_name IN ('minmax_min', 'minmax_max');")
        rows = cur.fetchall()
    params = {}
    for name, vec_str in rows:
        params[name] = np.array(parse_vector_str(vec_str), dtype=np.float64)
    if 'minmax_min' not in params or 'minmax_max' not in params:
        raise RuntimeError("MinMax params missing. Run indexer first.")
    return params


def minmax_normalize(vector: List[float], min_vals: np.ndarray, max_vals: np.ndarray) -> List[float]:
    """Áp dụng Min-Max scaling: (v - min) / (max - min)."""
    v = np.array(vector, dtype=np.float64)
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1.0, range_vals)
    normed = (v - min_vals) / range_vals
    return normed.tolist()