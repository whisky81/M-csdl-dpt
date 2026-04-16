# config.py
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "audio_db",
    "user": "postgres",
    "password": "2324",
}

# Tham số trích xuất âm thanh
SR = 16000
HOP_LENGTH = 512
N_MFCC = 13
SILENCE_DB = -40
FEATURE_DIM = 18          # 5 hand-crafted + 13 MFCCs
FMIN = 65.41              # C2 ~ 65.41 Hz
FMAX = 2093.0             # C7 ~ 2093 Hz

# Silence detection
TOP_DB = 40               # ngưỡng dB để xác định silence (dùng với librosa.effects.split)

# Tìm kiếm
K_SEGMENTS_PER_QUERY = 20 # số segment gần nhất lấy cho mỗi segment truy vấn
MAX_CANDIDATE_FILES = 50  # số file ứng viên tối đa đưa vào DTW
TOP_K_FILES = 5           # số file trả về cuối cùng
# config.py
DEFAULT_ENGINE = "ivfflat"   # hoặc "kdtree"