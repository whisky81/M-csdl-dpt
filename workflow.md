```
Upload file
    │
    ▼
Phân đoạn + trích xuất raw vectors
    │
    ▼
Min‑Max normalize (dùng global min/max)
    │
    ▼
Chọn Engine:
    ├─ IVFFlat ──► Tìm K segment gần nhất bằng Euclidean (<->) qua pgvector
    │
    └─ KD‑Tree ──► Tìm K segment gần nhất bằng Euclidean qua cKDTree
    │
    ▼
Gom candidate files (đếm tần suất file_id)
    │
    ▼
Với mỗi candidate:
    - Lấy chuỗi minmax_vec
    - Tính Dynamic Time Warping (Euclidean cục bộ) với chuỗi truy vấn
    │
    ▼
Sắp xếp theo Dynamic Time Warping tăng dần → Top‑5
```