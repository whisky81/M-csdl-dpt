import os
import logging
import re
import tempfile
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, render_template, request, jsonify, send_from_directory # type: ignore
from werkzeug.utils import secure_filename # type: ignore

# Import các module từ dự án
from main import DBPool, _extract_worker, DB_CONFIG
from query import (
    search_ivfflat,
    search_kdtree,
    precision_recall,
    _load_norm_params,
    normalize_query,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Cấu hình Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB limit
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()  # Lưu file tạm

# Kết nối DB (dùng chung pool)
db_pool = DBPool(DB_CONFIG, minconn=2, maxconn=8)

# Cache tham số chuẩn hóa (load một lần, dùng lại)
NORM_PARAMS = None


def get_norm_params():
    """Load và cache normalization parameters từ DB."""
    global NORM_PARAMS
    if NORM_PARAMS is None:
        NORM_PARAMS = _load_norm_params(db_pool)
    return NORM_PARAMS


# ----------------------------------------------------------------------
# Route phục vụ file audio tĩnh từ đường dẫn trong DB
@app.route('/audio/<int:obj_id>')
def serve_audio(obj_id: int):
    """
    Trả về file audio dựa trên obj_id. Đường dẫn file được lấy từ DB.
    """
    with db_pool.conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_path FROM multimedia_objects WHERE obj_id = %s;", (obj_id,))
        row = cur.fetchone()
    if not row:
        return "File not found", 404
    file_path = Path(row[0])
    if not file_path.exists():
        return "File missing on disk", 404
    # Gửi file với tên gốc
    return send_from_directory(
        directory=file_path.parent,
        path=file_path.name,
        as_attachment=False,
        mimetype='audio/flac'
    )


# ----------------------------------------------------------------------
# Hàm thực hiện tất cả 6 tổ hợp tìm kiếm và trả về dict kết quả
def run_all_combos(
    file_path: str,
    query_speaker: Optional[str] = None,
    k: int = 5
) -> Dict[str, Dict]:
    """
    Thực hiện 6 tổ hợp tìm kiếm, trả về dict:
        key: tên tổ hợp (theo định dạng "norm | metric | index")
        value: {
            'results': list of (obj_id, speaker_name, distance, file_path),
            'precision': float,
            'recall': float
        }
    """
    # 1. Trích xuất raw features
    raw_vec = _extract_worker(file_path)
    if raw_vec is None:
        raise RuntimeError("Feature extraction failed")

    # 2. Chuẩn hóa query
    params = get_norm_params()
    mm_vec, zs_vec = normalize_query(raw_vec, params)

    # 3. Auto-detect speaker nếu file đã có trong DB
    if query_speaker is None:
        with db_pool.conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT speaker_name FROM multimedia_objects WHERE file_path = %s;",
                (file_path,)
            )
            row = cur.fetchone()
        if row:
            query_speaker = row[0] or ""
        else:
            query_speaker = None

    # 4. Định nghĩa các tổ hợp (theo đúng thứ tự ưu tiên)
    combos = [
        # (label, search_fn, kwargs, vec)
        ("min-max | euclidean | ivfflat",
         search_ivfflat,
         {"norm": "minmax", "metric": "euclidean", "db": db_pool, "k": k},
         mm_vec),
        ("min-max | cosine    | ivfflat",
         search_ivfflat,
         {"norm": "minmax", "metric": "cosine",    "db": db_pool, "k": k},
         mm_vec),
        ("z-score | euclidean | ivfflat",
         search_ivfflat,
         {"norm": "zscore", "metric": "euclidean", "db": db_pool, "k": k},
         zs_vec),
        ("z-score | cosine    | ivfflat",
         search_ivfflat,
         {"norm": "zscore", "metric": "cosine",    "db": db_pool, "k": k},
         zs_vec),
        ("min-max | euclidean | kd-tree",
         search_kdtree,
         {"norm": "minmax", "k": k},
         mm_vec),
        ("z-score | euclidean | kd-tree",
         search_kdtree,
         {"norm": "zscore", "k": k},
         zs_vec),
    ]

    results_dict = {}

    for label, search_fn, kwargs, vec in combos:
        # Tìm kiếm
        results = search_fn(vec, **kwargs)   # list of (obj_id, speaker, distance)

        # Lấy thêm file_path cho mỗi kết quả
        enriched = []
        with db_pool.conn() as conn:
            cur = conn.cursor()
            for obj_id, spk, dist in results:
                cur.execute("SELECT file_path FROM multimedia_objects WHERE obj_id = %s;", (obj_id,))
                row = cur.fetchone()
                file_path_result = row[0] if row else None
                enriched.append((obj_id, spk, dist, file_path_result))

        # Tính Precision/Recall (nếu có speaker)
        if query_speaker:
            # results đã có (obj_id, speaker, dist)
            prec, rec = precision_recall(
                results,
                query_speaker=query_speaker,
                query_file_path=file_path,
                db=db_pool
            )
        else:
            prec, rec = None, None

        results_dict[label] = {
            'results': enriched,
            'precision': prec,
            'recall': rec
        }

    return results_dict


# ----------------------------------------------------------------------
# Routes
@app.route('/')
def index():
    """Trang chính với form upload."""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """
    Xử lý file upload, chạy tất cả tổ hợp, trả về JSON kết quả.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Lưu file tạm
    filename = secure_filename(file.filename)
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(tmp_path)
    
    # option get speaker name from file name 
    parts = filename.split('-')
    query_speaker = parts[0] if len(parts) == 3 and re.sub(r'\s+', '', parts[0]).isalpha() else None
        

    try:
        # Gọi hàm xử lý tất cả tổ hợp
        all_results = run_all_combos(tmp_path, query_speaker=query_speaker, k=5)

        # Chuẩn bị dữ liệu JSON
        response = {}
        for combo_label, data in all_results.items():
            response[combo_label] = {
                'results': [
                    {
                        'obj_id': obj_id,
                        'speaker': speaker,
                        'distance': distance,
                        'file_url': f'/audio/{obj_id}' if file_path else None
                    }
                    for obj_id, speaker, distance, file_path in data['results']
                ],
                'precision': data['precision'],
                'recall': data['recall']
            }

        return jsonify(response)

    except Exception as e:
        log.exception("Search error")
        return jsonify({'error': str(e)}), 500
    finally:
        # Xóa file tạm
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ----------------------------------------------------------------------
# Khởi chạy
if __name__ == '__main__':
    # Đảm bảo các bảng và index đã có (chạy main.py trước đó)
    app.run(debug=True, host='0.0.0.0', port=5000)