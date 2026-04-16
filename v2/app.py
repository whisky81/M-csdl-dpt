# app.py
import os
import tempfile
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from db import DBPool
from extractor import process_file
from normalizer import load_minmax_params, minmax_normalize
from search import search_top_files

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

db_pool = DBPool(minconn=2, maxconn=8)

# Cache normalization params
norm_params = None

def get_norm_params():
    global norm_params
    if norm_params is None:
        norm_params = load_minmax_params(db_pool)
    return norm_params


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Lấy lựa chọn engine từ form
    engine = request.form.get('engine', 'ivfflat')      # 'ivfflat' hoặc 'kdtree'
    # metric = request.form.get('metric', 'euclidean')    # chỉ dùng cho ivfflat

    filename = secure_filename(file.filename)
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(tmp_path)

    try:
        # 1. Phân đoạn và trích xuất vector thô
        seg_results = process_file(tmp_path)
        if not seg_results:
            return jsonify({'error': 'Could not extract segments from audio'}), 400

        raw_vectors = [vec for _, _, _, vec in seg_results]

        # 2. Chuẩn hóa Min-Max
        params = get_norm_params()
        min_vals = params['minmax_min']
        max_vals = params['minmax_max']
        norm_vectors = [minmax_normalize(v, min_vals, max_vals) for v in raw_vectors]

        # 3. Tìm kiếm theo engine được chọn
        use_ivfflat = (engine == 'ivfflat')
        results = search_top_files(
            db_pool,
            norm_vectors,
            use_ivfflat=use_ivfflat,
            metric="euclidean"
        )

        # 4. Bổ sung URL audio cho frontend
        for r in results:
            r['audio_url'] = f"/audio/{r['file_id']}"

        return jsonify({'results': results})

    except Exception as e:
        log.exception("Search error")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.route('/audio/<int:file_id>')
def serve_audio(file_id):
    """Phục vụ file audio gốc."""
    with db_pool.conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT file_path FROM voices_metadata WHERE file_id = %s;", (file_id,))
        row = cur.fetchone()
        if not row:
            return "File not found", 404
        file_path = row[0]
    if not os.path.exists(file_path):
        return "File missing on disk", 404
    from flask import send_file
    return send_file(file_path, mimetype='audio/flac')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)