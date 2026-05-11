# HUONG DAN THAY DOI - He thong CSDL Tim kiem Giong noi Nam

> **Ngay cap nhat**: 2026-05-11
> **Phien ban**: v2.1

---

## 1. Tong quan thay doi

Cac cai tien duoc thuc hien dua tren yeu cau trong `tmp.md`, bao gom 5 nhom chinh:

| # | Nhom thay doi | File bi anh huong | Trang thai |
|:---|:---|:---|:---:|
| 1 | Giai thich 18 chieu dac trung | `v2/FEATURES.md` (MOI) | Done |
| 2 | Tim kiem dua tren metadata | `v2/search.py`, `v2/app.py` | Done |
| 3 | Giao dien Web cai tien | `v2/templates/index.html` | Done |
| 4 | Bo loc do dai (duration filter) | `v2/search.py`, `v2/app.py` | Done |
| 5 | Hien thi ket qua trung gian | `v2/search.py`, `v2/app.py`, `v2/templates/index.html` | Done |
| 6 | So do khoi va quy trinh | `workflow.md` | Done |
| 7 | Endpoint thong ke CSDL | `v2/app.py` (MOI) | Done |

---

## 2. Chi tiet tung thay doi

### 2.1. FEATURES.md — Giai thich 18 chieu dac trung

**File moi**: `v2/FEATURES.md`

Noi dung:

- **5 dac trung thu cong**: avg_energy (E), avg_zcr (Z), silence_ratio (S), avg_pitch (P), avg_centroid (C)
- **13 he so MFCC**: M1 - M13 (Mel-Frequency Cepstral Coefficients)
- **Phan loai**: Thuoc tinh nao the hien su **tuong dong** (cung nguoi noi -> gia tri gan nhau) va thuoc tinh nao the hien su **khac biet** (khac nguoi noi -> gia tri khac nhau)
- **Ma tran danh gia**: Moi dac trung duoc cho diem tu 1-5 sao ve kha nang the hien tuong dong / khac biet
- **Ly do lua chon**: 5 ly do co so (dai dien 2 mien thoi gian/tan so, sinh ly giong noi, thang Mel mo phong tai nguoi, hieu qua tinh toan, phu hop DTW)
- **Gia tri thong tin**: Xep hang tu rat cao den thap

**Cach doc**: Mo file `v2/FEATURES.md` — day la tai lieu tham khao chinh cho cau hoi "thuoc tinh nao the hien su tuong dong/khac biet giua cac giong noi".

---

### 2.2. Tim kiem Metadata

**File thay doi**: `v2/search.py` (them ham `search_metadata`), `v2/app.py` (them route `/metadata-search`)

**Chuc nang moi**:
- Tim kiem file am thanh dua tren cac truong metadata: `speaker_name`, `duration_seconds`, `word_count`, `file_size_bytes`
- Khong can upload file am thanh — chi can nhap bo loc tren Web UI
- Ho tro tim kiem tuong doi (ILIKE) cho ten nguoi noi
- Ho tro phan trang (limit, offset)

**API moi**: `GET /metadata-search`

Tham so (query string):

| Tham so | Kieu | Mo ta |
|:---|:---|:---|
| `speaker` | string | Tim kiem tuong doi ten nguoi noi |
| `duration_min` | float | Do dai toi thieu (giay) |
| `duration_max` | float | Do dai toi da (giay) |
| `word_count_min` | int | So tu toi thieu |
| `word_count_max` | int | So tu toi da |
| `file_size_min` | int | Kich thuoc file toi thieu (bytes) |
| `file_size_max` | int | Kich thuoc file toi da (bytes) |
| `limit` | int | So ket qua moi trang (mac dinh 50) |
| `offset` | int | Vi tri bat dau (mac dinh 0) |

Vi du:
```
GET /metadata-search?speaker=John&duration_min=4&duration_max=7&limit=20
```

---

### 2.3. Bo loc Duration cho Tim kiem Am thanh

**File thay doi**: `v2/search.py` (cap nhat `search_top_files`), `v2/app.py` (cap nhat route `/search`)

**Chuc nang moi**: Khi tim kiem giong noi, co the loc ket qua theo do dai file (giay).

- Them 2 input `duration_min` va `duration_max` trong form upload
- Loc candidate files truoc khi tinh DTW — giup giam thoi gian tinh toan
- Phu hop yeu cau "cac file co cung do dai" trong de bai (loc file trong khoang [4, 7] giay)

---

### 2.4. Hien thi Ket qua Trung gian (Intermediate Steps)

**File thay doi**: `v2/search.py`, `v2/app.py`, `v2/templates/index.html`

**Chuc nang moi**: Sau khi tim kiem, he thong tra ve cac buoc trung gian de nguoi dung hieu ro qua trinh xu ly:

| Buoc | Ten truong | Y nghia |
|:---|:---|:---|
| 1 | `query_segments_count` | So segment query trich xuat duoc |
| 2 | `total_segment_matches` | Tong segment khop tim duoc |
| 2 | `per_segment_match_count` | So segment khop cho moi segment query |
| 2 | `engine` | Engine su dung (ivfflat/kdtree) |
| 3 | `candidate_files_count` | So file ung vien |
| 4 | `dtw_computed` | So DTW da tinh |
| 5 | `top_k` | So file tra ve |
| 5 | `dtw_details` | Chi tiet DTW cho top K |

Hien thi trong Web UI o panel "Ket qua trung gian".

---

### 2.5. Giao dien Web cai tien

**File thay doi**: `v2/templates/index.html`

**Thay doi**:
- **3 Tab**: "Tim kiem giong noi" | "Tim kiem Metadata" | "Thong ke CSDL"
- Panel **Ket qua trung gian** trong tab Tim kiem giong noi
- Form **Tim kiem Metadata** voi day du bo loc (speaker, duration, word count, file size)
- Phan trang cho ket qua metadata
- Panel **Thong ke CSDL** tu dong tai du lieu tu API `/stats`

---

### 2.6. Endpoint Thong ke CSDL

**File thay doi**: `v2/app.py` (them route `/stats`)

**API moi**: `GET /stats`

Tra ve:
- `total_files`: Tong so file (1329)
- `total_segments`: Tong so segment
- `duration_min/max/avg`: Do dai ngan nhat, dai nhat, trung binh
- `files_duration_4_7`: So file co do dai trong [4, 7] giay
- `distinct_speakers`: So nguoi noi phan biet
- `top_speakers`: Top 20 nguoi noi co nhieu file nhat

---

### 2.7. workflow.md — So do khoi va Quy trinh

**File thay doi**: `workflow.md` (viet lai hoan toan)

Noi dung moi:
- So do khoi he thong (Data Layer -> Indexing -> Search Engine)
- Quy trinh tong the 3 buoc: Chuan bi du lieu -> Indexing -> Tim kiem
- Quy trinh trich xuat dac trung (Feature Extraction Pipeline)
- Quy trinh DTW (Dynamic Time Warping) — minh hoa bang ma tran
- Bang mo ta cac ket qua trung gian
- Bang mo ta cac bang du lieu PostgreSQL
- Cong thuc tinh Euclidean distance va DTW distance

---

## 3. Cau truc file hien tai

```
M-csdl-DPT/
├── tmp.md                          # Yeu cau + tasks
├── workflow.md                     # So do khoi + quy trinh (DA CAP NHAT)
├── understand.md                   # Kien thuc nen tang (KHONG DOI)
├── CHANGES.md                      # File nay - huong dan thay doi
│
└── v2/
    ├── FEATURES.md                 # MOI - Giai thich 18 chieu dac trung
    ├── README.md                   # Huong dan chay (KHONG DOI)
    ├── app.py                      # DA CAP NHAT - them /metadata-search, /stats, duration filter, intermediate
    ├── config.py                   # KHONG DOI
    ├── db.py                       # KHONG DOI
    ├── extractor.py                # KHONG DOI
    ├── indexer.py                  # KHONG DOI
    ├── insert.py                   # KHONG DOI
    ├── normalizer.py               # KHONG DOI
    ├── search.py                   # DA CAP NHAT - them search_metadata(), duration filter, intermediate
    ├── requirements.txt            # KHONG DOI
    ├── metadata.csv                # KHONG DOI
    ├── kdtree_segments_minmax.pkl  # KHONG DOI (tu sinh boi indexer)
    └── templates/
        └── index.html              # DA CAP NHAT - 3 tab, metadata search, intermediate
```

---

## 4. Huong dan chay lai he thong

### Khong can chay lai indexing

Neu ban da chay `indexer.py` truoc do, cac thay doi khong anh huong den du lieu da index. Chi can:

```bash
cd v2
python app.py
```

Truy cap `http://localhost:5000` de su dung giao dien moi.

### Neu muon chay lai tu dau

```bash
# 1. Cai dat thu vien
pip install -r requirements.txt

# 2. Import metadata vao PostgreSQL
python insert.py
# => Chay lenh \copy trong psql

# 3. Indexing (lan dau hoac rebuild)
python indexer.py

# 4. Chay web server
python app.py
```

---

## 5. Cach doc cac file

| Muon biet... | Doc file... |
|:---|:---|
| Giai thich 18 dac trung, tuong dong vs khac biet | `v2/FEATURES.md` |
| So do khoi, quy trinh tong the | `workflow.md` |
| Kien thuc nen tang ve am thanh, DTW, MFCC | `understand.md` |
| Cach cai dat va chay | `v2/README.md` |
| Chi tiet API endpoints | Doc code `v2/app.py` (co comment tieng Anh) |
| Giao dien Web | Mo `http://localhost:5000` |
| Cac thay doi so voi phien ban cu | `CHANGES.md` (file nay) |

---

## 6. Luu y

- **18 chieu dac trung duoc giu nguyen**, khong thay doi so luong hay y nghia
- `search_top_files()` thay doi **kieu tra ve** tu `List[Dict]` thanh `Tuple[List[Dict], Dict]` — phan tu thu 2 la intermediate results
- `search_metadata()` la ham moi — khong anh huong den tim kiem giong noi hien co
- File `metadata.csv` khong thay doi
- Cac file `extractor.py`, `config.py`, `db.py`, `normalizer.py`, `insert.py`, `indexer.py` khong thay doi
