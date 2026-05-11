# app.py
import os
import tempfile
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from db import DBPool
from extractor import process_file
from normalizer import load_minmax_params, minmax_normalize
from search import search_top_files, search_metadata

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


# =====================================================================
#  AUDIO SIMILARITY SEARCH
# =====================================================================

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Lay lua chon engine tu form
    engine = request.form.get('engine', 'ivfflat')       # 'ivfflat' hoac 'kdtree'

    # Duration filter (optional)
    dur_min = request.form.get('duration_min', None)
    dur_max = request.form.get('duration_max', None)
    duration_min = float(dur_min) if dur_min else None
    duration_max = float(dur_max) if dur_max else None

    filename = secure_filename(file.filename)
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(tmp_path)

    try:
        # 1. Phan doan va trich xuat vector tho
        seg_results = process_file(tmp_path)
        if not seg_results:
            return jsonify({'error': 'Could not extract segments from audio'}), 400

        raw_vectors = [vec for _, _, _, vec in seg_results]

        # 2. Chuan hoa Min-Max
        params = get_norm_params()
        min_vals = params['minmax_min']
        max_vals = params['minmax_max']
        norm_vectors = [minmax_normalize(v, min_vals, max_vals) for v in raw_vectors]

        # 3. Tim kiem theo engine duoc chon
        use_ivfflat = (engine == 'ivfflat')
        results, intermediate = search_top_files(
            db_pool,
            norm_vectors,
            use_ivfflat=use_ivfflat,
            metric="euclidean",
            duration_min=duration_min,
            duration_max=duration_max,
        )

        # 4. Bo sung URL audio cho frontend
        for r in results:
            r['audio_url'] = f"/audio/{r['file_id']}"

        return jsonify({
            'results': results,
            'intermediate': intermediate
        })

    except Exception as e:
        log.exception("Search error")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# =====================================================================
#  METADATA SEARCH
# =====================================================================

@app.route('/metadata-search', methods=['GET'])
def metadata_search():
    """Tim kiem file dua tren metadata (khong can upload audio)."""
    speaker = request.args.get('speaker', None)
    dur_min = request.args.get('duration_min', None)
    dur_max = request.args.get('duration_max', None)
    wc_min = request.args.get('word_count_min', None)
    wc_max = request.args.get('word_count_max', None)
    fs_min = request.args.get('file_size_min', None)
    fs_max = request.args.get('file_size_max', None)
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    try:
        results, total = search_metadata(
            db_pool,
            speaker_name=speaker,
            duration_min=float(dur_min) if dur_min else None,
            duration_max=float(dur_max) if dur_max else None,
            word_count_min=int(wc_min) if wc_min else None,
            word_count_max=int(wc_max) if wc_max else None,
            file_size_min=int(fs_min) if fs_min else None,
            file_size_max=int(fs_max) if fs_max else None,
            limit=limit,
            offset=offset,
        )

        # Bo sung audio URL
        for r in results:
            r['audio_url'] = f"/audio/{r['file_id']}"

        return jsonify({
            'results': results,
            'total': total,
            'limit': limit,
            'offset': offset,
        })

    except Exception as e:
        log.exception("Metadata search error")
        return jsonify({'error': str(e)}), 500


# =====================================================================
#  STATS & DIAGNOSTICS
# =====================================================================

@app.route('/stats', methods=['GET'])
def stats():
    """Tra ve thong ke co ban ve CSDL."""
    try:
        with db_pool.conn() as conn:
            cur = conn.cursor()

            # Tong so file
            cur.execute("SELECT COUNT(*) FROM voices_metadata;")
            total_files = cur.fetchone()[0]

            # Tong so segment
            cur.execute("SELECT COUNT(*) FROM segments;")
            total_segments = cur.fetchone()[0]

            # Duration stats
            cur.execute("""
                SELECT MIN(duration_seconds), MAX(duration_seconds), AVG(duration_seconds)
                FROM voices_metadata;
            """)
            dmin, dmax, davg = cur.fetchone()

            # Count files in [4, 7] range
            cur.execute("SELECT COUNT(*) FROM voices_metadata WHERE duration_seconds BETWEEN 4 AND 7;")
            count_4_7 = cur.fetchone()[0]

            # Distinct speakers
            cur.execute("SELECT COUNT(DISTINCT speaker_name) FROM voices_metadata;")
            distinct_speakers = cur.fetchone()[0]

            # Speaker distribution
            cur.execute("""
                SELECT speaker_name, COUNT(*) as cnt
                FROM voices_metadata
                GROUP BY speaker_name
                ORDER BY cnt DESC
                LIMIT 20;
            """)
            top_speakers = [{"speaker": r[0], "count": r[1]} for r in cur.fetchall()]

        return jsonify({
            "total_files": total_files,
            "total_segments": total_segments,
            "duration_min": float(dmin) if dmin else 0,
            "duration_max": float(dmax) if dmax else 0,
            "duration_avg": float(davg) if davg else 0,
            "files_duration_4_7": count_4_7,
            "distinct_speakers": distinct_speakers,
            "top_speakers": top_speakers,
        })
    except Exception as e:
        log.exception("Stats error")
        return jsonify({'error': str(e)}), 500


# =====================================================================
#  AUDIO SERVING
# =====================================================================

@app.route('/audio/<int:file_id>')
def serve_audio(file_id):
    """Phuc vu file audio goc."""
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
