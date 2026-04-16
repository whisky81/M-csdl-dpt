# search.py
import logging
import pickle
from collections import Counter
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from fastdtw import fastdtw

from db import DBPool, parse_vector_str
from normalizer import load_minmax_params, minmax_normalize
from config import K_SEGMENTS_PER_QUERY, MAX_CANDIDATE_FILES, TOP_K_FILES, FEATURE_DIM

log = logging.getLogger(__name__)

KD_TREE_PICKLE = "kdtree_segments_minmax.pkl"


def load_kdtree():
    """Tải cKDTree và mapping từ file pickle."""
    with open(KD_TREE_PICKLE, "rb") as f:
        tree, mapping = pickle.load(f)
    return tree, mapping

def search_segments_ivfflat(db: DBPool, query_vec: List[float], k: int = 20) -> List[Tuple[int, int, float]]:
    """
    Tìm k segment gần nhất dùng IVFFlat (pgvector) với Euclidean distance (<->).
    Trả về list (segment_id, file_id, distance).
    """
    vec_str = "[" + ",".join(map(str, query_vec)) + "]"
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("SET ivfflat.probes = 10;")
        cur.execute("""
            SELECT s.segment_id, s.file_id, s.minmax_vec <-> %s::vector AS distance
            FROM segments s
            ORDER BY distance
            LIMIT %s;
        """, (vec_str, k))
        rows = cur.fetchall()
    return [(r[0], r[1], float(r[2])) for r in rows]

def search_segments_kdtree(query_vec: List[float], k: int = 20) -> List[Tuple[int, int, float]]:
    """Tìm k segment gần nhất dùng cKDTree (Euclidean)."""
    tree, mapping = load_kdtree()
    q = np.array(query_vec, dtype=np.float64)
    distances, indices = tree.query(q, k=k)
    if k == 1:
        distances = [distances]
        indices = [indices]
    results = []
    for dist, idx in zip(distances, indices):
        seg_id, file_id, _ = mapping[idx]
        results.append((seg_id, file_id, float(dist)))
    return results


def get_candidate_files(segment_results: List[Tuple[int, int, float]]) -> List[int]:
    """
    Từ danh sách segment khớp (có thể trùng file_id), đếm tần suất file_id,
    trả về danh sách file_id xuất hiện nhiều nhất (tối đa MAX_CANDIDATE_FILES).
    """
    file_counter = Counter(fid for _, fid, _ in segment_results)
    return [fid for fid, _ in file_counter.most_common(MAX_CANDIDATE_FILES)]


def get_segment_sequence(db: DBPool, file_id: int) -> List[List[float]]:
    """Lấy chuỗi minmax_vec của một file, sắp xếp theo segment_index."""
    with db.conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT minmax_vec::text FROM segments
            WHERE file_id = %s
            ORDER BY segment_index;
        """, (file_id,))
        rows = cur.fetchall()
    return [parse_vector_str(r[0]) for r in rows]


def dtw_distance(seq1: List[List[float]], seq2: List[List[float]]) -> float:
    """
    Tính khoảng cách DTW giữa hai chuỗi vector.
    Sử dụng Euclidean làm metric cục bộ.
    """
    if not seq1 or not seq2:
        return float('inf')
    # fastdtw trả về (distance, path)
    dist, _ = fastdtw(seq1, seq2, dist=lambda x, y: np.linalg.norm(np.array(x)-np.array(y)))
    return dist


def search_top_files(db: DBPool, query_segments: List[List[float]], use_ivfflat: bool = True, metric: str = "euclidean") -> List[Dict]:
    """
    Pipeline tìm kiếm chính:
      - Với mỗi segment query, tìm K_SEGMENTS_PER_QUERY segment gần nhất.
      - Gom candidate files.
      - Tính DTW cho từng candidate, sắp xếp theo DTW tăng dần.
      - Trả về TOP_K_FILES file kèm metadata và DTW distance.
    """
    all_segment_matches = []
    for qvec in query_segments:
        if use_ivfflat:
            matches = search_segments_ivfflat(db, qvec, k=K_SEGMENTS_PER_QUERY)
        else:
            matches = search_segments_kdtree(qvec, k=K_SEGMENTS_PER_QUERY)
        all_segment_matches.extend(matches)

    candidate_file_ids = get_candidate_files(all_segment_matches)
    if not candidate_file_ids:
        return []

    log.info("Candidates: %d files", len(candidate_file_ids))

    # Tính DTW cho từng candidate
    results = []
    for fid in candidate_file_ids:
        cand_seq = get_segment_sequence(db, fid)
        dtw_dist = dtw_distance(query_segments, cand_seq)
        results.append((fid, dtw_dist))

    # Sắp xếp theo DTW tăng dần
    results.sort(key=lambda x: x[1])
    top = results[:TOP_K_FILES]

    # Lấy thông tin metadata để trả về
    output = []
    with db.conn() as conn:
        cur = conn.cursor()
        for fid, dist in top:
            cur.execute("""
                SELECT file_name, speaker_name, file_path, duration_seconds
                FROM voices_metadata WHERE file_id = %s;
            """, (fid,))
            row = cur.fetchone()
            if row:
                output.append({
                    "file_id": fid,
                    "file_name": row[0],
                    "speaker": row[1],
                    "file_path": row[2],
                    "duration": row[3],
                    "dtw_distance": dist,
                })
    return output