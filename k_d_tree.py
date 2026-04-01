import pickle
from scipy.spatial import cKDTree # type: ignore
from pathlib import Path
from main import DBPool, _vec_to_list

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "mmdb",
    "user": "postgres",
    "password": "2324",
}

def build_kd_tree(points, map_obj_id):
    """
    Xây dựng hoặc tải cây KD-tree cùng với mapping obj_id.
    - points: mảng 2D (N x 18) các vector đặc trưng.
    - map_obj_id: danh sách obj_id theo đúng thứ tự của points.
    Trả về (kdtree, mapping).
    """
    path = Path('kdtree_with_map.pickle')
    if path.exists():
        with open(path, 'rb') as f:
            kdtree, saved_map = pickle.load(f)
        # Kiểm tra mapping có khớp không (ví dụ so sánh độ dài)
        # if saved_map == map_obj_id:
        #     print("Loaded existing KD-tree.")
        #     return kdtree, saved_map
        # else:
        #     print("Mapping changed, rebuilding KD-tree.")
        return kdtree, saved_map
    # Xây dựng mới
    kdtree = cKDTree(points, leafsize=16)
    with open(path, 'wb') as f:
        pickle.dump((kdtree, map_obj_id), f)
    print("Built and saved new KD-tree.")
    return kdtree, map_obj_id

def sim(query_point, k=5):
    kdtree, mapping = build_kd_tree([], [])

    # Truy vấn
    distances, indices = kdtree.query(query_point, k=k)
    print("Nearest neighbors (obj_id, distance):")
    for idx, dist in zip(indices, distances):
        print(f"obj_id: {mapping[idx]}, distance: {dist}")

def main():
    db = DBPool(DB_CONFIG, minconn=1, maxconn=10)
    try:
        with db.conn() as conn:
            cur = conn.cursor()
        cur.execute("""
            SELECT obj_id, min_max_scaling_vec
            FROM multimedia_objects;
        """)
        rows = cur.fetchall()
        map_obj_id = [row[0] for row in rows]
        points = [_vec_to_list(row[1]) for row in rows]

        kdtree, mapping = build_kd_tree(points, map_obj_id)

        # Lấy query vector từ obj_id = 1
        cur.execute("SELECT min_max_scaling_vec FROM multimedia_objects WHERE obj_id = 1;")
        query_vec = cur.fetchone()[0]
        query_point = _vec_to_list(query_vec)

        distances, indices = kdtree.query(query_point, k=6)
        print("Nearest neighbors (obj_id, distance):")
        for idx, dist in zip(indices, distances):
            print(f"obj_id: {mapping[idx]}, distance: {dist}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if db:
            db.close()
            
if __name__ == "__main__":
    main()