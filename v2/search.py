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
    """Tai cKDTree va mapping tu file pickle."""
    with open(KD_TREE_PICKLE, "rb") as f:
        tree, mapping = pickle.load(f)
    return tree, mapping

def search_segments_ivfflat(db: DBPool, query_vec: List[float], k: int = 20) -> List[Tuple[int, int, float]]:
    """
    Tim k segment gan nhat dung IVFFlat (pgvector) voi Euclidean distance (<->).
    Tra ve list (segment_id, file_id, distance).
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
    """Tim k segment gan nhat dung cKDTree (Euclidean)."""
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
    Tu danh sach segment khop (co the trung file_id), dem tan suat file_id,
    tra ve danh sach file_id xuat hien nhieu nhat (toi da MAX_CANDIDATE_FILES).
    """
    file_counter = Counter(fid for _, fid, _ in segment_results)
    return [fid for fid, _ in file_counter.most_common(MAX_CANDIDATE_FILES)]


def get_segment_sequence(db: DBPool, file_id: int) -> List[List[float]]:
    """Lay chuoi minmax_vec cua mot file, sap xep theo segment_index."""
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
    Tinh khoang cach DTW giua hai chuoi vector.
    Su dung Euclidean lam metric cuc bo.
    """
    if not seq1 or not seq2:
        return float('inf')
    # fastdtw tra ve (distance, path)
    dist, _ = fastdtw(seq1, seq2, dist=lambda x, y: np.linalg.norm(np.array(x)-np.array(y)))
    return dist


# =====================================================================
#  METADATA SEARCH
# =====================================================================

def search_metadata(
    db: DBPool,
    speaker_name: str = None,
    duration_min: float = None,
    duration_max: float = None,
    word_count_min: int = None,
    word_count_max: int = None,
    file_size_min: int = None,
    file_size_max: int = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple:
    """
    Tim kiem file am thanh dua tren metadata (khong dung vector).
    Ho tro loc theo: speaker_name, duration_seconds, word_count, file_size_bytes.

    Tra ve (results: list, total: int)
    """
    conditions = []
    params = []

    if speaker_name:
        conditions.append("v.speaker_name ILIKE %s")
        params.append(f"%{speaker_name}%")

    if duration_min is not None:
        conditions.append("v.duration_seconds >= %s")
        params.append(duration_min)

    if duration_max is not None:
        conditions.append("v.duration_seconds <= %s")
        params.append(duration_max)

    if word_count_min is not None:
        conditions.append("v.word_count >= %s")
        params.append(word_count_min)

    if word_count_max is not None:
        conditions.append("v.word_count <= %s")
        params.append(word_count_max)

    if file_size_min is not None:
        conditions.append("v.file_size_bytes >= %s")
        params.append(file_size_min)

    if file_size_max is not None:
        conditions.append("v.file_size_bytes <= %s")
        params.append(file_size_max)

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    with db.conn() as conn:
        cur = conn.cursor()

        # Dem tong so ket qua
        cur.execute(
            f"SELECT COUNT(*) FROM voices_metadata v WHERE {where_clause};",
            params
        )
        total = cur.fetchone()[0]

        # Lay ket qua phan trang
        cur.execute(
            f"""
            SELECT v.file_id, v.file_name, v.file_path, v.speaker_name,
                   v.file_size_bytes, v.word_count, v.duration_seconds
            FROM voices_metadata v
            WHERE {where_clause}
            ORDER BY v.file_id
            LIMIT %s OFFSET %s;
            """,
            params + [limit, offset]
        )
        rows = cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "file_id": r[0],
            "file_name": r[1],
            "file_path": r[2],
            "speaker": r[3] or "unknown",
            "file_size_bytes": r[4],
            "word_count": r[5],
            "duration_seconds": r[6],
        })

    return results, total


# =====================================================================
#  AUDIO SIMILARITY SEARCH
# =====================================================================

def search_top_files(
    db: DBPool,
    query_segments: List[List[float]],
    use_ivfflat: bool = True,
    metric: str = "euclidean",
    duration_min: float = None,
    duration_max: float = None,
) -> tuple:
    """
    Pipeline tim kiem chinh:
      - Tim K segment gan nhat cho moi segment query.
      - Gom candidate files theo tan suat xuat hien.
      - Loc candidate theo duration (neu co).
      - Tinh DTW cho tung candidate, sap xep theo DTW tang dan.
      - Tra ve TOP_K_FILES file kem metadata, DTW distance, va intermediate steps.
    """
    intermediate = {}

    # Step 1: Thong tin query
    intermediate["query_segments_count"] = len(query_segments)
    intermediate["query_vector_dim"] = len(query_segments[0]) if query_segments else 0

    # Step 2: Tim segment khop
    all_segment_matches = []
    per_segment_matches = []
    for qvec in query_segments:
        if use_ivfflat:
            matches = search_segments_ivfflat(db, qvec, k=K_SEGMENTS_PER_QUERY)
        else:
            matches = search_segments_kdtree(qvec, k=K_SEGMENTS_PER_QUERY)
        all_segment_matches.extend(matches)
        per_segment_matches.append(len(matches))

    intermediate["total_segment_matches"] = len(all_segment_matches)
    intermediate["per_segment_match_count"] = per_segment_matches
    intermediate["engine"] = "ivfflat" if use_ivfflat else "kdtree"

    # Step 3: Gom candidate files
    candidate_file_ids = get_candidate_files(all_segment_matches)

    # Step 4: Loc duration (neu co)
    if (duration_min is not None or duration_max is not None) and candidate_file_ids:
        with db.conn() as conn:
            cur = conn.cursor()
            duration_conds = []
            duration_params = []
            if duration_min is not None:
                duration_conds.append("duration_seconds >= %s")
                duration_params.append(duration_min)
            if duration_max is not None:
                duration_conds.append("duration_seconds <= %s")
                duration_params.append(duration_max)
            placeholders = ",".join(["%s"] * len(candidate_file_ids))
            cur.execute(
                f"SELECT file_id FROM voices_metadata WHERE file_id IN ({placeholders}) AND {' AND '.join(duration_conds)};",
                candidate_file_ids + duration_params
            )
            valid_ids = {r[0] for r in cur.fetchall()}
            candidate_file_ids = [fid for fid in candidate_file_ids if fid in valid_ids]

    intermediate["candidate_files_count"] = len(candidate_file_ids)

    if not candidate_file_ids:
        intermediate["dtw_computed"] = 0
        return [], intermediate

    log.info("Candidates after filter: %d files", len(candidate_file_ids))

    # Step 5: Tinh DTW cho tung candidate
    results = []
    dtw_details = []
    for fid in candidate_file_ids:
        cand_seq = get_segment_sequence(db, fid)
        dtw_dist = dtw_distance(query_segments, cand_seq)
        results.append((fid, dtw_dist))
        dtw_details.append({
            "file_id": fid,
            "dtw_distance": round(dtw_dist, 4),
            "cand_segments": len(cand_seq),
        })

    intermediate["dtw_computed"] = len(results)

    # Step 6: Sap xep va chon Top K
    results.sort(key=lambda x: x[1])
    top = results[:TOP_K_FILES]
    intermediate["top_k"] = len(top)
    intermediate["dtw_details"] = sorted(dtw_details, key=lambda x: x["dtw_distance"])[:TOP_K_FILES]

    # Lay thong tin metadata
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
    return output, intermediate
