from pathlib import Path
import numpy as np
from k_d_tree import sim
from main import _extract_worker, DBPool, _vec_to_list
# from pgvector.psycopg2 import register_vector
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mmdb",
    "user": "postgres",
    "password": "2324",
}


def cmp(vec1, vec2):
    if not vec1 or not vec2:
        return False
    if len(vec1) != len(vec2):
        return False
    for a, b in zip(vec1, vec2):
        if abs(a - b) > 1e-6:
            return False
    return True


def vec_by_id(obj_id, idx=4):
    db = DBPool(DB_CONFIG, minconn=1, maxconn=10)
    row = None
    try:
        with db.conn() as conn:
            cur = conn.cursor()
        cur.execute(
            """SELECT * FROM multimedia_objects WHERE obj_id = %s;""", (obj_id,)
        )
        row = cur.fetchone()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if db:
            db.close()
    return _vec_to_list(row[idx]) if row else None


def normalize(vec):
    db = DBPool(DB_CONFIG, minconn=1, maxconn=10)
    res = None
    try:
        with db.conn() as conn:
            cur = conn.cursor()
        cur.execute("""SELECT * FROM normalization_params;""")
        rows = cur.fetchall()[:2]
        vec_np = np.array(vec, dtype=np.float64)
        min_v = np.array(_vec_to_list(rows[0][1]), dtype=np.float64)
        max_v = np.array(_vec_to_list(rows[1][1]), dtype=np.float64)
        range_v = max_v - min_v
        range_safe = np.where(range_v == 0, 1.0, range_v)

        res = (vec_np - min_v) / range_safe
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if db:
            db.close()
    return res


def main():
    path = Path("/home/a/static/LibriSpeech/dev-clean/251_137823_251-137823-0016.flac")
    if not path.exists():
        print("FileNotFound")
        return
    raw_feature_vec = _extract_worker(str(path))
    query_point = _vec_to_list(normalize(raw_feature_vec))
    # print(min_max_vec)

    print("min-max + euclidean + [ivfflat]")
    try:
        db = DBPool(DB_CONFIG, minconn=1, maxconn=10)
        with db.conn() as conn:
            # register_vector(conn)
            cur = conn.cursor()
        vec_str = "[" + ",".join(map(str, query_point)) + "]"
        cur.execute("""
            SET ivfflat.probes = 6;
            SELECT obj_id, speaker_name,
                    min_max_scaling_vec <-> %s::vector AS distance
            FROM multimedia_objects
            ORDER BY distance
            LIMIT %s;""", (vec_str,10,))
        rows = cur.fetchall()
        for obj_id, speaker_name, dist in rows:
            print(f"obj_id: {obj_id}, speaker_name: {speaker_name}, distance: {dist}")
    finally:
        if db:
            db.close()

    print("min-max + euclidean + [kdtree]")
    sim(query_point, k=10)


if __name__ == "__main__":
    main()
