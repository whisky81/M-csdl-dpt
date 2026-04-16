# Hệ CSDL Lưu trữ và Tìm kiếm Giọng nói Đàn ông

Đây là bài tập lớn môn Cơ sở dữ liệu đa phương tiện. Hệ thống thu thập file âm thanh giọng nam, trích rút đặc trưng âm học, lưu vào PostgreSQL và cho phép tìm kiếm file âm thanh tương đồng nhất với một file đầu vào bất kỳ.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Cấu trúc dự án](#2-cấu-trúc-dự-án)
3. [Cài đặt môi trường](#3-cài-đặt-môi-trường)
4. [Thiết lập PostgreSQL](#4-thiết-lập-postgresql)
5. [Chạy pipeline (theo đúng thứ tự)](#5-chạy-pipeline-theo-đúng-thứ-tự)
6. [Chạy web demo](#6-chạy-web-demo)
7. [Kiến trúc kỹ thuật](#7-kiến-trúc-kỹ-thuật)
8. [Câu hỏi thường gặp](#8-câu-hỏi-thường-gặp)

---

## 1. Tổng quan hệ thống

```
File âm thanh (.flac/.wav)
        │
        ▼
  Trích rút đặc trưng âm học (18 chiều)
  [Energy, ZCR, Silence Ratio, Pitch, Spectral Centroid, 13×MFCC]
        │
        ├── Min-Max Scaling → lưu vào cột min_max_scaling_vec
        └── Z-Score Normalization → lưu vào cột z_score_nor_vec
                │
                ├── ivfflat (pgvector) → tìm kiếm gần đúng (ANN), nhanh
                └── cKDTree (scipy)    → tìm kiếm chính xác, lưu ra file .pkl
```

**6 tổ hợp tìm kiếm** mà hệ thống hỗ trợ:

| # | Chuẩn hóa | Khoảng cách | Cấu trúc index |
|---|-----------|-------------|----------------|
| 1 | Min-Max   | Euclidean   | ivfflat        |
| 2 | Min-Max   | Cosine      | ivfflat        |
| 3 | Z-Score   | Euclidean   | ivfflat        |
| 4 | Z-Score   | Cosine      | ivfflat        |
| 5 | Min-Max   | Euclidean   | cKDTree        |
| 6 | Z-Score   | Euclidean   | cKDTree        |

> **Tại sao không có cKDTree + Cosine?** scipy `cKDTree` chỉ hỗ trợ chuẩn Minkowski (bao gồm Euclidean). Cosine similarity cần cấu trúc dữ liệu khác nên bỏ qua tổ hợp này.

Kết quả trả về **top-5** file giống nhất, kèm **Precision@5** và **Recall@5** dựa trên `speaker_name`.

---

## 2. Cấu trúc dự án

```
project/
├── main.py          # Trích rút đặc trưng + chuẩn hóa + lưu DB
├── k_d_tree.py      # Build cây KD-tree và lưu ra file pickle
├── query.py         # Chạy thử 6 tổ hợp trên 1 file cụ thể (CLI)
├── app.py           # Web demo (Flask)
├── requirements.txt
├── README.md
├── kdtree_minmax.pkl  # (tự sinh ra sau khi chạy k_d_tree.py)
└── kdtree_zscore.pkl  # (tự sinh ra sau khi chạy k_d_tree.py)
```

---

## 3. Cài đặt môi trường

```bash
# 1. Tạo virtual environment
python3 -m venv .venv

# 2. Kích hoạt
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate.bat

# 3. Cài thư viện
pip install -r requirements.txt
```

> Nếu gặp lỗi khi cài hàng loạt, cài từng gói: `pip install <tên_gói>`

---

## 4. Thiết lập PostgreSQL

### 4.1 Tạo bảng
`Lưu ý`: tạo database tên `mmdb` trước rồi sau đó tải extension `pgvector` theo hướng dẫn ở [pgvector repo](https://github.com/pgvector/pgvector)

Kết nối vào PostgreSQL và chạy các lệnh sau:

```sql
-- Bật extension pgvector (hỗ trợ lưu và tính khoảng cách vector)
CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng chính lưu metadata + vector đặc trưng
CREATE TABLE multimedia_objects (
    obj_id               SERIAL PRIMARY KEY,
    file_name            TEXT NOT NULL,
    file_path            TEXT NOT NULL UNIQUE,  -- đường dẫn tuyệt đối trên máy
    speaker_name         TEXT,                  -- dùng để tính Precision/Recall
    raw_feature_vector   VECTOR(18),            -- vector gốc chưa chuẩn hóa
    min_max_scaling_vec  VECTOR(18),            -- sau khi Min-Max scaling
    z_score_nor_vec      VECTOR(18)             -- sau khi Z-Score normalization
);
```

Bảng `normalization_params` **không cần tạo thủ công** — file `main.py` tự tạo khi chạy.

### 4.2 Insert dữ liệu thô

Insert ít nhất 3 cột `file_name`, `file_path`, `speaker_name` vào bảng. Ví dụ với dataset LibriSpeech:

```sql
INSERT INTO multimedia_objects (file_name, file_path, speaker_name)
VALUES
  ('251-137823-0000.flac', '/home/a/static/LibriSpeech/dev-clean/251-137823-0000.flac', 'John Rose'),
  ('652-129742-0000.flac', '/home/a/static/LibriSpeech/dev-clean/652-129742-0000.flac', 'dexter');
  -- ... thêm các file khác
```

> `speaker_name` là người nói . Hệ thống dùng trường này làm **ground truth** để tính Precision/Recall — file của cùng một người là "kết quả đúng".
> `speaker_name` tìm thấy trong file `SPEAKERS.TXT` khi tải dữ liệu âm thanh về

### 4.3 Tạo ivfflat index (sau khi đã có đủ dữ liệu)

> **Quan trọng:** Tạo index sau khi đã insert xong toàn bộ dữ liệu và chạy `main.py`, vì ivfflat dùng K-Means để tìm tâm cụm — cần có dữ liệu mới build được.

```sql
-- lists = 36 ≈ sqrt(1329), probes nên dùng trong khoảng [6, 10]

CREATE INDEX idx_minmax_euclidean ON multimedia_objects
USING ivfflat (min_max_scaling_vec vector_l2_ops) WITH (lists = 36);

CREATE INDEX idx_minmax_cosine ON multimedia_objects
USING ivfflat (min_max_scaling_vec vector_cosine_ops) WITH (lists = 36);

CREATE INDEX idx_zscore_euclidean ON multimedia_objects
USING ivfflat (z_score_nor_vec vector_l2_ops) WITH (lists = 36);

CREATE INDEX idx_zscore_cosine ON multimedia_objects
USING ivfflat (z_score_nor_vec vector_cosine_ops) WITH (lists = 36);
```

---

## 5. Chạy pipeline (theo đúng thứ tự)

### Bước 1 — `main.py`: trích rút đặc trưng + chuẩn hóa

```bash
python main.py
```

File này làm các việc sau:
- Đọc danh sách file từ DB (`file_path`)
- Dùng `librosa` trích rút vector 18 chiều từ mỗi file `.flac`
- Lưu `raw_feature_vector` vào DB
- Tính `min`, `max`, `mean`, `std` trên toàn bộ dataset → lưu vào bảng `normalization_params`
- Tính `min_max_scaling_vec` và `z_score_nor_vec` → cập nhật vào DB

Sau khi chạy xong, bảng `normalization_params` sẽ có đúng 4 dòng:

| param_name    | ý nghĩa                          |
|---------------|----------------------------------|
| `min_max_min` | Giá trị nhỏ nhất mỗi chiều       |
| `min_max_max` | Giá trị lớn nhất mỗi chiều       |
| `z_score_mean`| Trung bình mỗi chiều             |
| `z_score_std` | Độ lệch chuẩn mỗi chiều          |

> Nếu DB đã có `raw_feature_vector` từ lần chạy trước, `main.py` bỏ qua bước trích rút và chỉ chạy lại chuẩn hóa.

---

### Bước 2 — Tạo ivfflat index

Chạy 4 câu SQL ở mục [4.3](#43-tạo-ivfflat-index-sau-khi-đã-có-đủ-dữ-liệu) trong psql hoặc pgAdmin.

---

### Bước 3 — `k_d_tree.py`: build cây KD-tree

```bash
python k_d_tree.py
```

Sinh ra 2 file pickle:
- `kdtree_minmax.pkl` — cây KD-tree xây trên `min_max_scaling_vec`
- `kdtree_zscore.pkl` — cây KD-tree xây trên `z_score_nor_vec`

Mỗi file lưu cả cây lẫn **mapping** `index → (obj_id, speaker_name)`, dùng khi trả kết quả.

> Nếu 2 file pickle đã tồn tại, chạy lại sẽ không build lại. Muốn build lại (ví dụ sau khi thêm dữ liệu mới): `build_trees(force=True)` trong code, hoặc xóa 2 file `.pkl` rồi chạy lại.

---

### Bước 4 — `query.py`: test thử trên CLI

Sửa biến `QUERY_FILE` ở cuối file `query.py` thành đường dẫn file muốn thử, rồi chạy:

```bash
python query.py
```

Output mẫu:

```
2026-04-02 10:11:30,127 [INFO] Connection pool created (min=1, max=6).
2026-04-02 10:11:30,127 [INFO] Extracting features from 652_129742_652-129742-0000.flac …
2026-04-02 10:11:33,696 [INFO] Speaker auto-detected from DB: 'Scott Walter'
2026-04-02 10:11:33,697 [INFO] KD-tree pickles already exist. Use force=True to rebuild.

============================================================
  Query file : 652_129742_652-129742-0000.flac
  Speaker    : Scott Walter
  Top-k      : 5
============================================================

────────────────────────────────────────────────────────────
  ivfflat | min-max  | euclidean
────────────────────────────────────────────────────────────
  #    obj_id   speaker                distance
  ──── ──────── ──────────────────── ──────────
  1    129      Scott Walter           0.000000
  2    29       Scott Walter           0.215878
  3    261      Scott Walter           0.223460
  4    172      Scott Walter           0.238310
  5    977      Scott Walter           0.246009

  Precision@5: 0.8000   Recall@5: 0.0704
```

> **Precision@5 = 0.8** nghĩa là 4/5 kết quả trả về đúng là giọng của speaker `652`.
> **Recall@5 = 0.1** nghĩa là hệ thống tìm được 4 file, trong khi tổng số file của speaker `652` trong DB là 40.

---

## 6. Chạy web demo

```bash
python app.py
# hoặc
export FLASK_APP=app.py
flask run
```

Truy cập `http://localhost:5000`, upload file `.flac` và xem kết quả trực tiếp trên giao diện web. Kết quả hiển thị cả 6 tổ hợp, kèm Precision/Recall và nút nghe thử từng file tìm được.

---

## 7. Kiến trúc kỹ thuật

### 7.1 Vector đặc trưng 18 chiều

| STT | Tên đặc trưng        | Miền    | Ý nghĩa ngắn gọn                         |
|-----|----------------------|---------|------------------------------------------|
| 1   | Average Energy       | Thời gian | Độ to/nhỏ của giọng                     |
| 2   | Zero-Crossing Rate   | Thời gian | Đặc trưng phụ âm, phân biệt giọng/nhạc  |
| 3   | Silence Ratio        | Thời gian | Nhịp điệu nói (ngắt nghỉ nhiều/ít)       |
| 4   | Pitch (F0)           | Tần số  | Tần số cơ bản — đặc trưng quan trọng nhất để nhận diện giọng nam |
| 5   | Spectral Centroid    | Tần số  | "Độ sáng" của giọng                      |
| 6–18| MFCC 1–13           | Tần số  | Mô phỏng cách tai người nghe âm thanh — nhận diện cá nhân |

### 7.2 Tại sao dùng 2 loại chuẩn hóa?

**Min-Max** đưa mọi giá trị về `[0, 1]` — phù hợp khi các chiều có đơn vị rất khác nhau (Pitch tính bằng Hz, Energy rất nhỏ).

**Z-Score** đưa về `mean=0, std=1` — phù hợp hơn cho cosine similarity vì loại bỏ ảnh hưởng của độ lớn tuyệt đối.

### 7.3 Tại sao dùng 2 loại index?

**ivfflat (pgvector):** Tìm kiếm gần đúng (ANN) dùng K-Means để chia không gian thành `lists=36` cụm. Khi query chỉ duyệt `probes=6` cụm gần nhất → **rất nhanh** với dataset lớn, nhưng có thể bỏ sót vài kết quả.

**cKDTree (scipy):** Tìm kiếm **chính xác 100%** bằng cách chia không gian theo trục. Chậm hơn ivfflat với dataset lớn nhưng đảm bảo kết quả tốt nhất có thể.

### 7.4 Precision & Recall được tính thế nào?

```
Ground truth = tất cả file trong DB có cùng speaker_name với query
               (trừ chính file query nếu đã có trong DB)

Precision@k = số file trả về đúng speaker / k
Recall@k    = số file trả về đúng speaker / tổng số file của speaker đó trong DB
```

---

## 8. Câu hỏi thường gặp

**Q: Chạy `main.py` bao lâu?**
Khoảng 15–20 phút cho 1329 file, vì dùng `ProcessPoolExecutor` chạy song song trên 8 core. Lần chạy lại sẽ bỏ qua bước extraction nếu đã có `raw_feature_vector`.

**Q: Lỗi `normalization_params has 0 rows`?**
Chưa chạy `main.py` hoặc `main.py` chạy lỗi giữa chừng. Chạy lại `main.py` đến khi thấy log `Pipeline complete.`

**Q: Lỗi `Could not load or build KD-tree`?**
Chưa chạy `k_d_tree.py`, hoặc DB chưa có dữ liệu cột `min_max_scaling_vec`. Chạy theo đúng thứ tự ở mục 5.

**Q: Tại sao Recall thấp dù Precision cao?**
Bình thường — `k=5` trong khi một speaker có thể có 40+ file trong DB. Recall@5 tối đa chỉ là `5/40 = 12.5%`. Tăng `k` để thấy Recall tăng lên.

**Q: Muốn test file không có trong DB?**
Đặt `query_speaker="tên_speaker"` khi gọi `run_all()` trong `query.py`. Hệ thống sẽ dùng tên đó để tính Precision/Recall thay vì tự detect từ DB.

**Q: Thêm dữ liệu mới thì làm gì?**
1. INSERT thêm vào `multimedia_objects` (chỉ cần 3 cột đầu)
2. Chạy lại `main.py` (sẽ chỉ extract file còn thiếu)
3. DROP và tạo lại 4 ivfflat index
4. Xóa 2 file `.pkl` và chạy lại `k_d_tree.py`
