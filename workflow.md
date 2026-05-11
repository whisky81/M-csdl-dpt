# So do khoi va Quy trinh He thong Tim kiem Giong noi Nam

## 1. So do khoi he thong (Block Diagram)

```
+-----------------------------------------------------------------------------+
|                           HE THONG TIM KIEM GIONG NOI NAM                   |
|                                                                             |
|  +------------------+     +------------------+     +----------------------+ |
|  | 1. DATA LAYER    |     | 2. INDEXING      |     | 3. SEARCH ENGINE     | |
|  |                  |     |                  |     |                      | |
|  | metadata.csv     |---->| insert.py        |     | app.py (Flask)       | |
|  | (1329 files)     |     | (clean + import) |     |  + /search           | |
|  |                  |     |                  |     |  + /metadata-search  | |
|  | voice-male-file/ |     | indexer.py       |     |  + /audio/<id>       | |
|  | (*.flac)         |     |  + extract       |     |  + /stats            | |
|  |                  |     |  + normalize     |     |                      | |
|  +--------+---------+     |  + ivfflat index |     | templates/index.html | |
|           |               |  + kd-tree       |     | (Web UI)             | |
|           v               +--------+---------+     +----------+-----------+ |
|  +------------------+              |                          |             |
|  | PostgreSQL +     |<-------------+                          |             |
|  | pgvector         |                                         |             |
|  |                  |<----------------------------------------+             |
|  | voices_metadata  |                                                       |
|  | segments (18D)   |                                                       |
|  | norm_params      |                                                       |
|  +------------------+                                                       |
+-----------------------------------------------------------------------------+
```

---

## 2. Quy trinh tong the

```
BUOC 0: CHUAN BI DU LIEU
  metadata.csv (1329 dong) + voice-male-file/ (1329 file .flac)
      |
      v
  insert.py: tao metadata_clean.csv -> \copy vao PostgreSQL
      |
      v
  voices_metadata: 1329 rows (file_name, file_path, speaker_name, ...)

BUOC 1: INDEXING (chay 1 lan)
  python indexer.py
      |
      +---> Phan doan am thanh (split by silence, top_db=40)
      |     1 file -> N segments (N thuong tu 1-10)
      |
      +---> Trich xuat vector 18D cho moi segment
      |     [E, Z, S, P, C, M1...M13]
      |
      +---> Luu vao bang segments (raw_vec, minmax_vec)
      |
      +---> Tinh global Min-Max -> luu normalization_params
      |
      +---> Cap nhat minmax_vec cho tat ca segments
      |
      +---> Tao IVFFlat index (pgvector)
      |
      +---> Xay dung KD-Tree -> kdtree_segments_minmax.pkl

BUOC 2: TIM KIEM (runtime)
  User upload file .flac qua Web UI
      |
      v
  app.py nhan request -> /search
      |
      v
  [Step 1] Phan doan file query -> raw vectors 18D
      |
      v
  [Step 2] Min-Max normalize (dung global params tu CSDL)
      |
      v
  [Step 3] Tim K segment gan nhat (IVFFlat hoac KD-Tree)
      |      Euclidean distance tren vector 18D
      |
      v
  [Step 4] Gom candidate files (tan suat file_id)
      |      + Loc duration (neu co bo loc)
      |
      v
  [Step 5] DTW cho tung candidate
      |      So khop chuoi segment query vs candidate
      |
      v
  [Step 6] Sap xep theo DTW -> Top 5
      |
      v
  Tra ve JSON: results + intermediate steps
```

---

## 3. Quy trinh trich xuat dac trung (Feature Extraction Pipeline)

```
File am thanh (*.flac)
      |
      v
librosa.load(sr=16000, mono=True)
      |  -> y: numpy array (16000 mau/giay)
      |  -> sr: 16000
      v
librosa.effects.split(y, top_db=40)
      |  -> intervals: [(start, end), ...]
      v
For each segment:
      |
      +---> [Time Domain]
      |     RMS Energy (E)     : librosa.feature.rms
      |     Zero Crossing (Z)  : librosa.feature.zero_crossing_rate
      |     Silence Ratio (S)  : frame_count(db < -40) / total_frames
      |
      +---> [Frequency Domain]
      |     Pitch (P)          : librosa.pyin (F0, 65-2093 Hz)
      |     Centroid (C)       : librosa.feature.spectral_centroid
      |
      +---> [Mel-Frequency]
            MFCC 1-13          : librosa.feature.mfcc(n_mfcc=13)
                                  -> mean over time axis

      => Vector 18D: [E, Z, S, P, C, M1, M2, ..., M13]
```

---

## 4. Quy trinh DTW (Dynamic Time Warping)

```
CHUOI QUERY: [v1, v2, v3]          (3 segment, 18D each)
CHUOI CANDIDATE: [w1, w2, w3, w4, w5]  (5 segment, 18D each)

      w1   w2   w3   w4   w5
  v1  d11  d12  d13  d14  d15
  v2  d21  d22  d23  d24  d25
  v3  d31  d32  d33  d34  d35

  dij = ||vi - wj||_2  (Euclidean distance)

DTW tim duong di tu (v1,w1) -> (v3,w5) toi thieu hoa tong dij.

Ket qua: DTW distance = tong dij tren duong toi uu.

DTW distance cang nho -> 2 file cang giong nhau.
```

---

## 5. Cac ket qua trung gian (Intermediate Results)

Khi thuc hien tim kiem, he thong tra ve cac thong tin trung gian:

| Buoc | Ten truong | Y nghia | Vi du |
|:---|:---|:---|:---|
| 1 | `query_segments_count` | So segment trich xuat duoc tu file query | 4 |
| 1 | `query_vector_dim` | So chieu cua moi vector | 18 |
| 2 | `total_segment_matches` | Tong so segment khop tim duoc | 80 |
| 2 | `per_segment_match_count` | So segment khop cho moi segment query | [20, 20, 20, 20] |
| 2 | `engine` | Engine duoc su dung | "ivfflat" |
| 3 | `candidate_files_count` | So file ung vien (sau loc) | 50 |
| 4 | `dtw_computed` | So file da tinh DTW | 50 |
| 5 | `top_k` | So file tra ve cuoi cung | 5 |
| 5 | `dtw_details` | Chi tiet DTW cho tung file top K | [{file_id, dtw_distance, cand_segments}, ...] |

---

## 6. Bang du lieu trong PostgreSQL

### voices_metadata

| Cot | Kieu | Mo ta |
|:---|:---|:---|
| `file_id` | SERIAL PK | Ma file |
| `file_name` | TEXT | Ten file (vd: 1272-128104-0000.flac) |
| `file_path` | TEXT UNIQUE | Duong dan tuyet doi |
| `speaker_name` | TEXT | Ten nguoi noi |
| `file_size_bytes` | BIGINT | Kich thuoc file (bytes) |
| `word_count` | INT | So tu trong transcript |
| `duration_seconds` | FLOAT | Do dai file (giay) |

### segments

| Cot | Kieu | Mo ta |
|:---|:---|:---|
| `segment_id` | SERIAL PK | Ma segment |
| `file_id` | INT FK | Ma file cha |
| `segment_index` | INT | Thu tu phan doan (0, 1, 2...) |
| `start_time` | FLOAT | Thoi gian bat dau (giay) |
| `end_time` | FLOAT | Thoi gian ket thuc (giay) |
| `raw_vec` | VECTOR(18) | Vector dac trung goc |
| `minmax_vec` | VECTOR(18) | Vector da chuan hoa Min-Max |

### normalization_params

| Cot | Kieu | Mo ta |
|:---|:---|:---|
| `param_name` | TEXT PK | 'minmax_min' hoac 'minmax_max' |
| `param_vector` | VECTOR(18) | Vector 18 chieu |

---

## 7. Cong thuc tinh khoang cach

### Euclidean Distance (L2)

```
d(a, b) = sqrt( SUM[(a_i - b_i)^2] )   for i = 1..18
```

Su dung cho:
- Tim segment gan nhat (IVFFlat, KD-Tree)
- Metric cuc bo trong DTW

### DTW Distance

```
DTW(Q, C) = min_path SUM[ d(q_i, c_j) ]
```

Trong do:
- Q: chuoi segment cua file query (M segments)
- C: chuoi segment cua file candidate (N segments)
- d(q_i, c_j): Euclidean distance giua 2 vector 18D
- min_path: duong di toi uu co the co gian thoi gian

### Ket qua cuoi cung

Top 5 file co DTW distance nho nhat, sap xep tang dan.
