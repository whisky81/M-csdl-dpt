
---

# **Nền tảng Toán học, Kiến trúc Hệ thống và Triển khai**

## Mục lục

1.  Giải thích Chủ đề (Toán học trước tiên)
2.  Các Khái niệm và Công thức Cốt lõi
3.  Kiến trúc Hệ thống
4.  Giải thích Mã nguồn
5.  Ánh xạ giữa Lý thuyết và Triển khai

---

# PHẦN 1: GIẢI THÍCH CHỦ ĐỀ (TOÁN HỌC TRƯỚC TIÊN)

## 1.1 Âm thanh như một Tín hiệu Vật lý

Âm thanh là sóng áp suất cơ học lan truyền trong không khí. Khi được số hóa, âm thanh trở thành một chuỗi rời rạc các mẫu biên độ:

**Tín hiệu liên tục:** x(t) ∈ ℝ, với t là thời gian (giây)

**Tín hiệu rời rạc:** x\[n\] = x(nT_s), trong đó T_s = 1/f_s là chu kỳ lấy mẫu

**Định lý lấy mẫu Nyquist-Shannon** phát biểu rằng để tái tạo hoàn hảo tín hiệu có tần số lớn nhất f_max, tốc độ lấy mẫu phải thỏa mãn:

f_s ≥ 2 · f_max

Đối với giọng nói con người (f_max ≈ 8.000 Hz), tốc độ lấy mẫu chuẩn là f_s = 16.000 Hz — hệ thống này sử dụng chính xác tốc độ này.

**Giải tích:**  
- Việc chọn f_s = 16 kHz gấp đôi băng tần giọng nói (8 kHz) đảm bảo không có hiện tượng chồng phổ (aliasing).  
- Giọng nam có tần số cơ bản (cao độ) thường trong khoảng **85 Hz – 255 Hz**, thấp hơn đáng kể so với giọng nữ (165 Hz – 255 Hz). Dải tần này là yếu tố phân biệt chính, vì vậy hệ thống đặt FMIN = 65,41 Hz (nốt C2) và FMAX = 2093,0 Hz (nốt C7) cho việc phát hiện cao độ.  
- Giá trị 65,41 Hz và 2093 Hz tương ứng với 7 quãng tám (từ C2 đến C7), bao phủ toàn bộ dải giọng nói của cả nam và nữ, đồng thời loại bỏ tạp âm tần số thấp (dưới 65 Hz) và họa âm bậc cao không cần thiết.

## 1.2 Xử lý Tín hiệu (Dạng sóng → Không gian Đặc trưng)

### Dạng sóng

Dạng sóng âm thanh thô là một vector N chiều của các mẫu PCM (Điều chế mã xung):

**x** = \[x\[0\], x\[1\], …, x\[N-1\]\] ∈ ℝᴺ

Với tệp âm thanh 5 giây tại f_s = 16.000 Hz: N = 80.000 mẫu.

### Biến đổi Fourier thời gian ngắn (STFT)

STFT phân tích nội dung tần số trên các cửa sổ (khung) chồng lấn ngắn. Cho hàm cửa sổ w\[n\] độ dài L:

STFT{x}(m, k) = Σₙ x\[n\] · w\[n - mH\] · e\^(-j2πkn/N_fft)

Trong đó:
- m = chỉ số khung
- k = chỉ số bin tần số
- H = bước nhảy (hop length) = 512 mẫu
- N_fft = kích thước FFT = 2048

**Phổ biên độ** là:
|STFT(m, k)|² = công suất tại khung m, bin tần số k

**Giải tích:**  
- Bước nhảy H = 512 mẫu tại 16 kHz tương đương 32 ms, tạo độ phân giải thời gian tốt cho giọng nói.  
- Kích thước FFT = 2048 cho độ phân giải tần số Δf = f_s / N_fft = 16000/2048 ≈ 7,8 Hz, đủ phân biệt các họa âm của giọng nói.  
- Sử dụng cửa sổ Hanning (mặc định trong librosa) để giảm rò rỉ phổ.

### Biến đổi Thang Mel

Thang Mel xấp xỉ nhận thức thính giác của con người. Tần số f_Hz được ánh xạ sang thang Mel như sau:

f_mel = 2595 · log₁₀(1 + f_Hz / 700)

Một bộ lọc gồm M = 40 bộ lọc hình tam giác chồng lấn được áp dụng lên phổ công suất:

S\[m, i\] = Σₖ |STFT(m, k)|² · H_i\[k\] với i = 1, …, M

Trong đó H_i\[k\] là bộ lọc Mel thứ i.

**Giải tích:**  
- Thang Mel nén các tần số cao (tai người ít nhạy hơn) và giãn tần số thấp (nhạy hơn).  
- 40 bộ lọc là giá trị phổ biến cho nhận dạng giọng nói; quá ít thì mất chi tiết, quá nhiều thì gây nhiễu và tăng chi phí tính toán.

### Hệ số Cepstral tần số Mel (MFCC)

MFCC trích xuất hình dạng đường thanh (bao phổ của giọng nói), lý tưởng cho nhận dạng người nói.

**Bước 1:** Áp dụng log cho năng lượng bộ lọc Mel:

log_S\[m, i\] = log(S\[m, i\])

**Bước 2:** Áp dụng Biến đổi Cosin rời rạc (DCT):

MFCC\[m, c\] = Σᵢ log_S\[m, i\] · cos(π·c·(i - 0.5)/M) với c = 1, …, N_mfcc

Hệ thống này trích xuất N_mfcc = 13 hệ số trên mỗi khung, sau đó lấy trung bình trên tất cả các khung:

**mfcc_avg** = (1/T) · Σₘ MFCC\[m, :\] ∈ ℝ¹³

**Giải tích:**  
- Logarit làm cho các thành phần nhân (bao phổ và chi tiết họa âm) trở thành tổng.  
- DCT giải tương quan các hệ số, tập trung năng lượng vào các hệ số đầu (quan trọng nhất).  
- Chỉ lấy 13 hệ số đầu vì các hệ số cao thường chứa nhiễu và ít thông tin phân biệt người nói.  
- Trung bình hóa theo thời gian giúp vector đặc trưng bất biến với độ dài âm thanh.

---

# PHẦN 2: CÁC KHÁI NIỆM VÀ CÔNG THỨC CỐT LÕI

## 2.1 Xây dựng Vector Đặc trưng

Với mỗi đoạn âm thanh, một **vector đặc trưng 18 chiều** được xây dựng:

**v** = \[E, ZCR, SR, P, SC, MFCC₁, MFCC₂, …, MFCC₁₃\] ∈ ℝ¹⁸

| **Chiều** | **Đặc trưng**            | **Công thức**                                               |
|-----------|--------------------------|-------------------------------------------------------------|
| 1         | Năng lượng trung bình (RMS) | E = sqrt((1/N) · Σ x\[n\]²)                               |
| 2         | Tỉ lệ qua không (ZCR)    | ZCR = (1/N) · Σ 1_{sign(x\[n\]) ≠ sign(x\[n-1\])}        |
| 3         | Tỉ lệ im lặng            | SR = count(RMS_dB < -40 dB) / total_frames                  |
| 4         | Cao độ trung bình (F0)   | P = mean(f0\[voiced_flag\]) qua thuật toán pYIN            |
| 5         | Trọng tâm phổ            | SC = Σₖ k · |X\[k\]|² / Σₖ |X\[k\]|²                      |
| 6–18      | MFCC 1–13                | MFCC_c (như đã trình bày)                                   |

**pYIN (Probabilistic YIN) cho Cao độ:**  
pYIN ước lượng tần số cơ bản bằng phần mở rộng xác suất của thuật toán YIN. Hàm sai phân d_τ cho độ trễ τ là:

d_τ = Σₙ (x\[n\] - x\[n+τ\])²

Hàm sai phân chuẩn hóa trung bình tích lũy:

d'_τ = { 1, nếu τ = 0; d_τ / ((1/τ) · Σⱼ₌₁\^τ dⱼ), nếu ngược lại }

Cao độ được tìm tại d'_τ giảm dưới ngưỡng (thường là 0,1), sau đó được tinh chỉnh theo xác suất.

**Giải tích:**  
- **Năng lượng RMS** phản ánh độ to của tín hiệu, giúp phân biệt các đoạn thì thầm (năng lượng thấp) và nói bình thường.  
- **Tỉ lệ qua không** (ZCR) cao thường ứng với âm hữu thanh (voiced) có nhiều dao động qua zero hơn so với âm vô thanh (unvoiced) như phụ âm.  
- **Tỉ lệ im lặng** (SR) cho biết một đoạn có chứa nhiều khoảng lặng hay không, hỗ trợ phân đoạn chính xác.  
- **Cao độ** (F0) là đặc trưng sinh trắc học mạnh nhất cho giọng nam/nữ.  
- **Trọng tâm phổ** là "trung bình trọng số" của các tần số, cho biết độ sáng của âm sắc.  
- **MFCC** mô tả hình dạng bộ máy phát âm (kích thước thanh quản, khoang miệng...).

## 2.2 Chuẩn hóa

### Chuẩn hóa Min-Max

Chuẩn hóa từng chiều đặc trưng độc lập về đoạn \[0, 1\]:

v̂ᵢ = (vᵢ - min_i) / (max_i - min_i)

Trong đó min_i = giá trị nhỏ nhất toàn cục của chiều i trên **TẤT CẢ** các đoạn trong cơ sở dữ liệu.

Vector này được lưu thành **minmax_vec** tách biệt với **raw_vec** thô.

**Giải tích:**  
- Chuẩn hóa đảm bảo các đặc trưng có thang đo khác nhau (ví dụ: năng lượng có thể từ 0 đến 1, cao độ từ 65 Hz đến 2000 Hz) không làm sai lệch khoảng cách Euclidean.  
- Lưu cả raw_vec giúp khi bổ sung dữ liệu mới, ta có thể tính lại min/max toàn cục mà không mất thông tin gốc.  
- Công thức tránh chia cho 0 bằng cách thay max - min bằng 1 nếu bằng 0.

## 2.3 Độ đo Tương tự

### Khoảng cách Euclidean (L2)

d_L2(**a**, **b**) = ||**a** - **b**||₂ = sqrt(Σᵢ (aᵢ - bᵢ)²)

Được sử dụng bởi chỉ mục IVFFlat trong pgvector (toán tử <->) và cKDTree.

### Độ tương tự Cosine

cos(**a**, **b**) = (**a** · **b**) / (||**a**|| · ||**b**||)

Khoảng cách Cosine = 1 - cos(**a**, **b**). Được dùng trong chỉ mục IVFFlat thứ cấp.

### Dynamic Time Warping (DTW)

Đối với các chuỗi có độ dài thay đổi, DTW tìm sự căn chỉnh phi tuyến tối ưu:

Cho hai chuỗi **Q** = \[q₁, q₂, …, qₘ\] và **C** = \[c₁, c₂, …, cₙ\] với mỗi phần tử là một vector đặc trưng:

DTW(**Q**, **C**) = min (Σ d(qᵢ, cⱼ)) trên tất cả các đường đi (warping paths) hợp lệ

Một đường đi W = {w₁, w₂, …, wₖ} phải thỏa mãn:
- Đơn điệu: i(wₜ) ≤ i(wₜ₊₁)
- Liên tục: i(wₜ₊₁) - i(wₜ) ≤ 1
- Biên: w₁ = (1,1), wₖ = (m,n)

Hệ thức truy hồi:

DTWcost(i, j) = d(qᵢ, cⱼ) + min{DTW(i-1, j), DTW(i, j-1), DTW(i-1, j-1)}

Độ phức tạp thời gian: O(m × n) — đó là lý do tại sao việc giảm ứng viên là rất quan trọng.

**FastDTW** xấp xỉ DTW trong thời gian O(N) bằng cách giới hạn đường đi tìm kiếm, đây là cách triển khai được sử dụng ở đây.

**Giải tích:**  
- DTW cho phép so sánh hai đoạn giọng nói cùng nội dung nhưng nói với tốc độ khác nhau (nhanh/chậm).  
- Ràng buộc đơn điệu và liên tục đảm bảo không đảo ngược thời gian và bước nhảy tối đa 1 khung.  
- FastDTW giảm độ phức tạp bằng cách tính DTW trên tỷ lệ thô trước, sau đó tinh chỉnh.

## 2.4 Chỉ mục Tìm kiếm Gần đúng Láng giềng Gần nhất IVFFlat

IVFFlat (Inverted File Flat) phân hoạch không gian vector thành L cụm (ô Voronoi). Trong quá trình tìm kiếm:
1. Tìm các tâm cụm *probes* gần nhất với vector truy vấn.
2. Chỉ tìm kiếm các vector nằm trong các cụm đó.
3. Trả về K láng giềng gần nhất xấp xỉ.

Đánh đổi: Độ chính xác (điều khiển bởi probes) so với Tốc độ. Hệ thống này sử dụng lists = sqrt(N_segments) ≈ 82 và ivfflat.probes = 10.

## 2.5 cây k-d

Cây k-d phân hoạch không gian k chiều bằng cách chia cắt siêu phẳng theo trục luân phiên:
- Thời gian xây dựng: O(N log N) cho N điểm trong k chiều
- Thời gian truy vấn: O(log N) trung bình, O(N) trong trường hợp xấu nhất
- Triển khai: scipy.spatial.cKDTree với leafsize=16

Cây k-d phục vụ như một chỉ mục dự phòng phía CPU (được lưu trong tệp .pkl).

**Giải tích:**  
- IVFFlat là chỉ mục chính vì nó tận dụng tối ưu của cơ sở dữ liệu PostgreSQL (phân tán, đồng thời).  
- cKDTree là fallback khi không dùng được pgvector (ví dụ: môi trường không có PostgreSQL).  
- Số lượng danh sách (lists) được chọn theo căn bậc hai của số đoạn là khuyến nghị của pgvector để cân bằng tốc độ và độ chính xác.

---

# PHẦN 3: KIẾN TRÚC HỆ THỐNG

## 3.1 Kiến trúc Mức cao

Hệ thống tuân theo kiến trúc **MIRS (Hệ thống Truy xuất Thông tin Đa phương tiện)** cổ điển với hai pha riêng biệt:

**PHA OFFLINE (Lập chỉ mục)**

```
Tệp âm thanh (FLAC/WAV)
       ↓
[Phân đoạn] → Phát hiện khoảng lặng, chia mỗi tệp thành N đoạn
       ↓
[Trích xuất đặc trưng] → vector 18 chiều mỗi đoạn (raw_vec)
       ↓
[Chuẩn hóa toàn cục] → Min-Max normalization → minmax_vec
       ↓
[Lưu trữ CSDL] → PostgreSQL + tiện ích pgvector
       ↓
[Xây dựng chỉ mục] → IVFFlat (pgvector) + cKDTree (pickle)
```

**PHA ONLINE (Tìm kiếm)**

```
Tệp âm thanh truy vấn
       ↓
[Phân đoạn + Trích xuất đặc trưng] → Các đoạn truy vấn Q₁, Q₂, …, Qₘ
       ↓
[Chuẩn hóa] → Dùng min/max toàn cục lưu trong bảng normalization_params
       ↓
[Giai đoạn 1: LỌC] → IVFFlat / cây k-d → Top-K đoạn cho mỗi đoạn truy vấn
       ↓
[Tập ứng viên] → Đếm số lần xuất hiện file_id → TỐI ĐA 50 tệp ứng viên
       ↓
[Giai đoạn 2: XẾP HẠNG LẠI] → Khoảng cách DTW cho mỗi tệp ứng viên (theo thứ tự các đoạn)
       ↓
[Kết quả] → Top 5 tệp sắp xếp theo DTW tăng dần
```

**Giải tích:**  
- Phân đoạn dựa trên phát hiện khoảng lặng giúp loại bỏ các phần không có giọng nói, giảm nhiễu.  
- Hai giai đoạn: lọc thô bằng ANN (nhanh, O(log N)), sau đó tinh chỉnh bằng DTW (chậm nhưng chính xác) là chiến lược chuẩn trong truy xuất thông tin đa phương tiện.  
- Giới hạn 50 ứng viên đảm bảo thời gian DTW chấp nhận được (với độ dài trung bình mỗi tệp khoảng 5-10 đoạn, 50 tệp → tối đa 500 cặp đoạn, mỗi cặp DTW O(L²) nhưng FastDTW giúp thực tế rất nhanh).

## 3.2 Sơ đồ Cơ sở dữ liệu

**Bảng: voices_metadata** — Một dòng cho mỗi tệp âm thanh

| Cột                     | Kiểu          | Mô tả                                      |
|-------------------------|---------------|--------------------------------------------|
| file_id                 | SERIAL PK     | Định danh tự tăng                         |
| file_name               | TEXT          | Tên tệp gốc                               |
| file_path               | TEXT UNIQUE   | Đường dẫn tuyệt đối trên đĩa              |
| speaker_name            | TEXT          | Nhãn người nói (từ CSV metadata)          |
| file_size_bytes         | BIGINT        | Kích thước tệp                            |
| duration_seconds        | FLOAT         | Thời lượng âm thanh                       |

**Bảng: segments** — Một dòng cho mỗi đoạn âm thanh

| Cột                     | Kiểu          | Mô tả                                               |
|-------------------------|---------------|-----------------------------------------------------|
| segment_id              | SERIAL PK     | Định danh đoạn                                      |
| file_id                 | INT FK        | Tham chiếu voices_metadata                          |
| segment_index           | INT           | Chỉ số thứ tự trong tệp (bắt buộc cho DTW)          |
| start_time              | FLOAT         | Thời gian bắt đầu đoạn (giây)                      |
| end_time                | FLOAT         | Thời gian kết thúc đoạn (giây)                     |
| raw_vec                 | VECTOR(18)    | Vector đặc trưng 18 chiều chưa chuẩn hóa            |
| minmax_vec              | VECTOR(18)    | Vector đã chuẩn hóa Min-Max (dùng cho lập chỉ mục)  |

**Bảng: normalization_params** — Tham số chuẩn hóa toàn cục

| param_name             | param_vector                  |
|------------------------|-------------------------------|
| minmax_min             | VECTOR(18) — giá trị nhỏ nhất toàn cục |
| minmax_max             | VECTOR(18) — giá trị lớn nhất toàn cục |

**Các chỉ mục:**
- idx_seg_minmax_l2: IVFFlat trên minmax_vec với khoảng cách L2
- idx_seg_minmax_cosine: IVFFlat trên minmax_vec với khoảng cách cosine
- idx_seg_file_id: B-tree trên file_id để JOIN nhanh
- idx_seg_file_segment: B-tree trên (file_id, segment_index) cho sắp xếp DTW

**Giải tích:**  
- Lưu cả raw_vec và minmax_vec giúp có thể tính lại chuẩn hóa khi thêm dữ liệu mới (vì min/max toàn cục có thể thay đổi).  
- Chỉ mục (file_id, segment_index) đảm bảo thứ tự các đoạn trong một tệp được lấy ra đúng, cần thiết cho DTW.  
- Hai chỉ mục IVFFlat (L2 và cosine) cho phép thử nghiệm hai độ đo khác nhau mà không cần xây dựng lại.

## 3.3 Đồ thị Phụ thuộc Module

```
app.py (Máy chủ Web Flask)
├── extractor.py ← librosa, numpy
├── normalizer.py ← db.py, numpy
├── search.py ← db.py, normalizer.py, fastdtw, scipy
└── db.py ← psycopg2, config.py

indexer.py (Tác vụ ngoại tuyến hàng loạt)
├── extractor.py
├── normalizer.py
└── db.py

config.py ← Không phụ thuộc (chỉ hằng số)
```

**Giải tích:**  
- Sự tách biệt rõ ràng: `extractor.py` chỉ làm biến đổi tín hiệu, `normalizer.py` chuẩn hóa, `search.py` tìm kiếm, `db.py` quản lý kết nối.  
- `indexer.py` chạy độc lập, có thể được gọi định kỳ hoặc khi có dữ liệu mới.  
- `app.py` là ứng dụng web, không chứa logic xử lý nặng, chỉ điều phối.

---

# PHẦN 4: GIẢI THÍCH MÃ NGUỒN

## 4.1 [[config.py]{.underline}](http://config.py) — Hằng số Hệ thống

Module cấu hình trung tâm định nghĩa tất cả các siêu tham số:

- **SR = 16000**: Tốc độ lấy mẫu (Hz), tuân thủ Nyquist cho giọng nói.
- **HOP_LENGTH = 512**: Bước nhảy khung cho STFT (~32ms tại 16kHz).
- **N_MFCC = 13**: Số hệ số MFCC.
- **FEATURE_DIM = 18**: Tổng kích thước vector đặc trưng (5 đặc trưng thủ công + 13 MFCC).
- **FMIN = 65,41 Hz, FMAX = 2093,0 Hz**: Dải phát hiện cao độ (C2 đến C7).
- **SILENCE_DB = -40**: Ngưỡng phân loại khung là im lặng (decibel).
- **TOP_DB = 40**: Ngưỡng cho phân đoạn dựa trên im lặng.
- **K_SEGMENTS_PER_QUERY = 20**: Số kết quả ANN trên mỗi đoạn truy vấn.
- **MAX_CANDIDATE_FILES = 50**: Kích thước tập ứng viên cho xếp hạng lại DTW.
- **TOP_K_FILES = 5**: Số kết quả cuối cùng trả về.

**Vấn đề bảo mật phát hiện:** Mật khẩu cơ sở dữ liệu được mã cứng ("password": "2324"). Cần chuyển sang biến môi trường.

**Giải tích:**  
- SILENCE_DB = -40 dB tương đương với biên độ rất nhỏ (gần như im lặng tuyệt đối), giúp phát hiện chính xác các khoảng dừng giữa các câu nói.  
- TOP_DB = 40 dB là ngưỡng phân đoạn tương đối cao, chỉ những phần có năng lượng lớn hơn mức nhiễu nền 40 dB mới được coi là có giọng nói.  
- K_SEGMENTS_PER_QUERY = 20: mỗi đoạn truy vấn (thường dài 1-2 giây) tìm 20 đoạn gần nhất; nếu truy vấn có 5 đoạn, tổng ứng viên thô là 100, sau khi gộp theo file có thể còn 50.

## 4.2 [[db.py]{.underline}](http://db.py) — Bộ kết nối Cơ sở dữ liệu

`DBPool` bao bọc `ThreadedConnectionPool` của psycopg2 cung cấp:
- Quản lý kết nối an toàn luồng (minconn=2, maxconn=8)
- Trình quản lý ngữ cảnh (conn()) với tự động commit/rollback/trả về
- Hàm tiện ích parse_vector_str() để giải tuần tự hóa chuỗi pgvector

**Lưu ý kiến trúc:** Bộ kết nối được khởi tạo một lần và chia sẻ qua các yêu cầu — đúng mẫu cho Flask.

**Giải tích:**  
- Sử dụng connection pool giúp tránh tạo kết nối mới mỗi request (tốn tài nguyên).  
- maxconn=8 phù hợp với Flask chạy đa luồng thông thường (mặc định 4 worker, mỗi worker 2 luồng).  
- Hàm parse_vector_str chuyển đổi chuỗi dạng "[0.1,0.2,...]" từ pgvector thành list Python.

## 4.3 [[extractor.py]{.underline}](http://extractor.py) — Trích xuất Đặc trưng

### segment_audio(file_path)

1.  Tải âm thanh với librosa tại SR=16000 Hz, mono.
2.  Gọi librosa.effects.split(y, top_db=40) để phát hiện các khoảng không im lặng.
3.  Trả về: danh sách các mảng numpy (dạng sóng) + danh sách các bộ (start_sec, end_sec).

### extract_features(audio, sr) — Lõi tính toán

Tính vector đặc trưng 18 chiều:

1.  **Năng lượng RMS**: librosa.feature.rms() → mean → avg_energy
2.  **ZCR**: librosa.feature.zero_crossing_rate() → mean → avg_zcr
3.  **Tỉ lệ im lặng**: tỷ lệ các khung RMS dưới SILENCE_DB (-40 dB)
4.  **Cao độ (pYIN)**: librosa.pyin() → chỉ các khung có giọng → mean f0 → avg_pitch
5.  **Trọng tâm phổ**: librosa.feature.spectral_centroid() → mean
6.  **13 MFCC**: librosa.feature.mfcc(n_mfcc=13) → trung bình theo thời gian

Trả về: [avg_energy, avg_zcr, silence_ratio, avg_pitch, avg_centroid] + avg_mfccs[0:13]

### process_file(file_path)

Điều phối phân đoạn + trích xuất, lọc các đoạn ngắn hơn HOP_LENGTH mẫu, trả về danh sách các bộ (audio_segment, start, end, feature_vector).

**Giải tích:**  
- Librosa xử lý âm thanh nền tảng; các hàm feature trả về ma trận (số khung × số đặc trưng), sau đó lấy trung bình theo khung để có vector cố định.  
- pYIN (librosa.pyin) là thuật toán mạnh mẽ cho giọng nói, đặc biệt khi có nhiễu nền.  
- Việc lọc các đoạn quá ngắn (dưới 512 mẫu ≈ 32 ms) tránh tạo các vector đặc trưng không ổn định.

## 4.4 [[normalizer.py]{.underline}](http://normalizer.py) — Chuẩn hóa Min-Max Toàn cục

### compute_global_minmax(db)

Lấy TẤT CẢ raw_vec từ bảng segments, xây dựng ma trận kích thước (N_segments × 18), tính min và max theo từng cột.

### save_minmax_params(db, min_vals, max_vals)

Lưu vector 18 chiều min và max vào bảng normalization_params.

### load_minmax_params(db)

Tải min/max đã lưu để sử dụng trong chuẩn hóa phía tìm kiếm.

### minmax_normalize(vector, min_vals, max_vals)

v̂ᵢ = (vᵢ - minᵢ) / max(maxᵢ - minᵢ, 1)

Chia cho 1 (thay vì 0) ngăn NaN khi một đặc trưng có phương sai bằng 0.

**Giải tích:**  
- Việc tính toán min/max toàn cục yêu cầu duyệt toàn bộ cơ sở dữ liệu (O(N)), do đó chỉ chạy một lần trong pha offline.  
- Khi thêm dữ liệu mới, cần chạy lại compute_global_minmax để cập nhật.  
- Bảo vệ max(range,1) rất quan trọng: nếu một đặc trưng không thay đổi (ví dụ tất cả các đoạn đều có ZCR = 0.5), thì max-min=0, nếu không xử lý sẽ gây lỗi chia cho 0.

## 4.5 [[indexer.py]{.underline}](http://indexer.py) — Lập chỉ mục ngoại tuyến

Indexer là pipeline chính offline, chạy một lần để xây dựng cơ sở dữ liệu.

### Các giai đoạn pipeline

**Giai đoạn 1: build_schema(db)** Tạo bảng với tiện ích pgvector nếu chưa tồn tại.

**Giai đoạn 2: index_all_files_parallel(db, rebuild, max_workers)**  
- Lấy tất cả file_path từ voices_metadata.  
- Khởi tạo ProcessPoolExecutor (tối đa min(cpu_count, 8) worker).  
- Mỗi worker gọi _process_file_worker(file_path) độc lập.  
- Kết quả được tập hợp bằng asyncio.gather(*futures).  
- Chèn hàng loạt các đoạn bằng execute_values (INSERT bulk hiệu quả).

**Giai đoạn 3: normalize_all_segments(db)**  
- Gọi compute_global_minmax() trên tất cả raw_vec.  
- Cập nhật cột minmax_vec cho mọi đoạn.

**Giai đoạn 4: create_ivfflat_indexes(db)**  
- Tính lists = max(10, int(sqrt(N))) theo hướng dẫn pgvector.  
- Tạo hai chỉ mục: L2 distance và cosine distance.

**Giai đoạn 5: build_kdtree(db)**  
- Tải tất cả vector đã chuẩn hóa vào bộ nhớ dưới dạng mảng numpy.  
- Xây dựng cKDTree(points, leafsize=16).  
- Lưu thành kdtree_segments_minmax.pkl để tải nhanh sau này.

**Vấn đề kiến trúc:** index_all_files_parallel sử dụng asyncio + ProcessPoolExecutor, nhưng build_schema và chèn đoạn là các lời gọi DB đồng bộ bên trong hàm async — một mẫu đồng bộ/bất đồng bộ hỗn hợp có thể gây lỗi.

**Giải tích:**  
- Xử lý song song bằng ProcessPoolExecutor hữu ích khi trích xuất đặc trưng (tốn CPU), nhưng cần chú ý đến GIL của Python.  
- Bulk INSERT (execute_values) nhanh hơn nhiều so với INSERT từng dòng, đặc biệt với hàng nghìn đoạn.  
- Lưu cKDTree dưới dạng pickle giúp khởi động nhanh mà không cần xây dựng lại cây mỗi lần.

## 4.6 [[search.py]{.underline}](http://search.py) — Truy xuất hai giai đoạn

### Giai đoạn 1A: search_segments_ivfflat(db, query_vec, k=20)

Thực hiện truy vấn ANN pgvector:

```sql
SET ivfflat.probes = 10;
SELECT segment_id, file_id, minmax_vec <-> '[ ... ]'::vector AS distance
FROM segments ORDER BY distance LIMIT 20;
```

Toán tử <-> tính khoảng cách L2. probes=10 nghĩa là 10 ô Voronoi được tìm kiếm (đánh đổi độ chính xác/tốc độ).

### Giai đoạn 1B: search_segments_kdtree(query_vec, k=20)

Tải tệp .pkl và gọi tree.query(q, k=20) trả về khoảng cách Euclidean và chỉ số, ánh xạ ngược lại (segment_id, file_id) qua danh sách ánh xạ đã lưu.

### get_candidate_files(segment_results)

Sử dụng Counter để đếm số lần xuất hiện file_id trên tất cả các kết quả đoạn truy vấn. Các tệp xuất hiện trong nhiều danh sách kết quả đoạn sẽ được xếp hạng cao hơn. Trả về tối đa MAX_CANDIDATE_FILES = 50 file ID.

### Giai đoạn 2: Xếp hạng lại DTW

Với mỗi tệp ứng viên:
1.  get_segment_sequence(db, file_id) — lấy chuỗi minmax_vec đã sắp xếp.
2.  dtw_distance(query_segments, cand_seq) — gọi fastdtw() với độ đo local Euclidean.

Kết quả được sắp xếp theo DTW tăng dần; top 5 được trả về cùng metadata.

**Giải tích:**  
- SET ivfflat.probes = 10 là câu lệnh điều chỉnh thời gian chạy: probes càng lớn càng chính xác nhưng chậm. Giá trị 10 là cân bằng tốt cho 82 cụm.  
- Việc đếm tần suất file_id dựa trên giả định rằng nếu một tệp có nhiều đoạn khớp với các đoạn truy vấn thì tệp đó có khả năng cao là kết quả đúng.  
- FastDTW (từ thư viện fastdtw) giảm độ phức tạp từ O(L²) xuống O(L) bằng cách co giãn đường đi ở độ phân giải thấp trước.

## 4.7 [[app.py]{.underline}](http://app.py) — Máy chủ Web Flask

**Các route:**
- GET / — Cung cấp giao diện HTML tìm kiếm
- POST /search — Nhận tệp âm thanh upload, trả về JSON kết quả
- GET /audio/<file_id> — Phát trực tiếp tệp âm thanh gốc (FLAC)

**Luồng endpoint search:**
1. Lưu tệp upload vào thư mục tạm hệ thống
2. Gọi process_file() để phân đoạn + trích xuất đặc trưng
3. Tải tham số chuẩn hóa đã lưu trong cache (lazy-loaded, cached toàn cục)
4. Chuẩn hóa vector truy vấn
5. Gọi search_top_files() với engine đã chọn (ivfflat hoặc kdtree)
6. Thêm trường audio_url vào mỗi kết quả
7. Trả về JSON; dọn dẹp tệp tạm trong khối finally

**Vấn đề phát hiện:**
- norm_params là biến toàn cục — không an toàn luồng trong triển khai đa tiến trình.
- Không có giới hạn tốc độ (rate limiting) trên endpoint /search.
- Phát trực tiếp âm thanh sử dụng mimetype cứng 'audio/flac' — sẽ không hoạt động với tệp .wav hoặc .mp3.

**Giải tích:**  
- Lazy loading norm_params: lần đầu tiên tìm kiếm sẽ tải từ DB, các lần sau dùng cache, giảm tải DB.  
- Tuy nhiên, nếu dùng nhiều worker process (ví dụ gunicorn với workers >1), mỗi worker có cache riêng, không đồng bộ → cần dùng Redis hoặc bộ nhớ chia sẻ.  
- Việc không giới hạn tốc độ có thể dẫn đến tấn công DoS; nên thêm Flask-Limiter.

---

# PHẦN 5: ÁNH XẠ — LÝ THUYẾT VÀ TRIỂN KHAI

## 5.1 Xử lý Tín hiệu → Mã nguồn

| **Khái niệm Toán học**                           | **Triển khai**                                      | **Tệp**                                            | **Tham chiếu Mã**                         |
|--------------------------------------------------|-----------------------------------------------------|----------------------------------------------------|-------------------------------------------|
| Định lý lấy mẫu Nyquist: f_s ≥ 2·f_max          | SR = 16000 Hz                                       | [[config.py]{.underline}](http://config.py)       | SR = 16000                                |
| Phân tích khung STFT                             | Bước nhảy H = 512 mẫu                               | [[config.py]{.underline}](http://config.py)       | HOP_LENGTH = 512                          |
| Phát hiện im lặng (ngưỡng top_db)                | Chia âm thanh thành các đoạn không im lặng          | [[extractor.py]{.underline}](http://extractor.py) | librosa.effects.split(y, top_db=TOP_DB)   |
| Ước lượng cao độ pYIN                           | pYIN xác suất với biên FMIN/FMAX                    | [[extractor.py]{.underline}](http://extractor.py) | librosa.pyin(audio, fmin=FMIN, fmax=FMAX) |
| Năng lượng RMS: E = sqrt((1/N)·Σx\[n\]²)        | RMS mỗi khung, sau đó trung bình                    | [[extractor.py]{.underline}](http://extractor.py) | librosa.feature.rms() → np.mean()         |
| Tỉ lệ qua không (ZCR)                           | ZCR mỗi khung, sau đó trung bình                    | [[extractor.py]{.underline}](http://extractor.py) | librosa.feature.zero_crossing_rate()      |
| Trọng tâm phổ: Σk·\|X\[k\]\|² / Σ\|X\[k\]\|²      | librosa.feature.spectral_centroid()                | [[extractor.py]{.underline}](http://extractor.py) | librosa.feature.spectral_centroid()       |
| Bộ lọc Mel + DCT (MFCC)                          | 13 hệ số MFCC, trung bình theo thời gian            | [[extractor.py]{.underline}](http://extractor.py) | librosa.feature.mfcc(n_mfcc=N_MFCC)       |

## 5.2 Trích xuất Đặc trưng → Lưu trữ Vector

| **Lý thuyết**                          | **Triển khai**                                        | **Quyết định thiết kế**                                |
|----------------------------------------|-------------------------------------------------------|--------------------------------------------------------|
| Vector đặc trưng 18 chiều mỗi đoạn     | VECTOR(18) trong PostgreSQL                           | Tiện ích pgvector cho phép lập chỉ mục ANN            |
| Vector mỗi đoạn (không phải mỗi tệp)   | Bảng segments với segment_index                       | Cho phép DTW nhận biết thứ tự chuỗi                    |
| Chuỗi có thứ tự cho DTW                | ORDER BY segment_index trong SQL                      | Đảm bảo tính toàn vẹn tuần tự                         |
| Biểu diễn kép (thô + chuẩn hóa)        | Cột raw_vec và minmax_vec                             | Raw để tái chuẩn hóa; minmax_vec cho ANN              |

## 5.3 Chuẩn hóa → Cơ sở dữ liệu

| **Lý thuyết**                                | **Triển khai**                                    | **Tệp**                                                    |
|----------------------------------------------|---------------------------------------------------|-------------------------------------------------------------|
| Min-Max toàn cục: v̂ = (v-min)/(max-min)      | Hàm minmax_normalize()                           | [[normalizer.py]{.underline}](http://normalizer.py)        |
| Min/max toàn cục tính trên TẤT CẢ vector     | Duyệt toàn bộ raw_vec                            | [[normalizer.py]{.underline}](http://normalizer.py): compute_global_minmax() |
| Lưu thống kê toàn cục                        | Bảng normalization_params                        | [[normalizer.py]{.underline}](http://normalizer.py): save_minmax_params() |
| Chuẩn hóa phía truy vấn với cùng tham số     | Tải tham số một lần, cache toàn cục              | [[app.py]{.underline}](http://app.py): get_norm_params()   |
| Bảo vệ phương sai bằng 0                     | Chia cho max(range, 1)                           | [[normalizer.py]{.underline}](http://normalizer.py): range_vals = np.where(...) |

## 5.4 Lập chỉ mục ANN → Chỉ mục Cơ sở dữ liệu

| **Lý thuyết**                                      | **Triển khai**                                       | **Hiệu năng**                                     |
|----------------------------------------------------|------------------------------------------------------|---------------------------------------------------|
| IVFFlat: phân hoạch thành L ô Voronoi              | CREATE INDEX USING ivfflat ... WITH (lists=82)       | sqrt(N) lists — thực hành tốt nhất pgvector       |
| Tham số probes điều khiển độ chính xác/tốc độ      | SET ivfflat.probes = 10                              | 10/82 ô được tìm kiếm mỗi truy vấn                |
| cây k-d cho không gian k chiều                     | scipy.spatial.cKDTree(points, leafsize=16)           | Dự phòng CPU, lưu dưới dạng pickle                |
| ANN trả về top-K xấp xỉ                           | K_SEGMENTS_PER_QUERY = 20                            | 20 ứng viên đoạn cho mỗi đoạn truy vấn           |

## 5.5 Tìm kiếm hai giai đoạn → Pipeline tìm kiếm

| **Lý thuyết**                                | **Triển khai**                                      | **Lý do**                                        |
|----------------------------------------------|-----------------------------------------------------|--------------------------------------------------|
| Giai đoạn 1: Bộ lọc xấp xỉ nhanh             | IVFFlat hoặc cây k-d ANN                            | O(log N) so với O(N) quét tuyến tính            |
| Tổng hợp ứng viên theo tần suất              | Counter(file_ids).most_common(50)                  | Tệp khớp nhiều đoạn truy vấn là ứng viên tốt   |
| Giai đoạn 2: Căn chỉnh chuỗi chính xác       | fastdtw(seq1, seq2, dist=euclidean)                | Xử lý độ dài thay đổi, căn chỉnh phi tuyến     |
| Kiểm soát độ phức tạp DTW O(N×M)             | MAX_CANDIDATE_FILES = 50                            | Giới hạn chặt chẽ ngăn bùng nổ bậc hai         |
| Xếp hạng cuối cùng                           | results.sort(key=lambda x: x[1])                    | Khoảng cách DTW tăng dần = giống nhất trước    |
| Trả về top-5 kết quả                         | TOP_K_FILES = 5                                     | Số lượng truy xuất chuẩn MIRS                  |

## 5.6 Đánh giá Chất lượng Kiến trúc

| **Hạng mục**           | **Điểm** | **Nhận xét**                                                                 |
|------------------------|----------|------------------------------------------------------------------------------|
| Tính đúng đắn toán học | A        | Công thức chính xác cho mọi đặc trưng; pipeline MFCC đúng                    |
| Thiết kế tìm kiếm 2 giai đoạn | A    | IVFFlat + DTW là mẫu sản xuất chuẩn                                           |
| Thiết kế cơ sở dữ liệu | A-       | Lưu trữ vector kép, ràng buộc khóa ngoại đúng, chỉ mục hợp lý                |
| Chất lượng mã nguồn    | B+       | Phân tách trách nhiệm tốt; có vài phản mẫu nhỏ trong xử lý bất đồng bộ        |
| Bảo mật                | C        | Mật khẩu DB mã cứng; không kiểm tra đầu vào về loại/kích thước tệp            |
| Sẵn sàng sản xuất      | B        | Thiếu: kiểm tra sức khỏe, giới hạn tốc độ, quản lý bí mật đúng cách          |
| Tài liệu               | B+       | Chú thích mã bằng tiếng Việt; docstring tốt trên các hàm chính               |

---

# PHỤ LỤC: BẢNG THUẬT NGỮ

| **Thuật ngữ**   | **Định nghĩa**                                                                                 |
|----------------|------------------------------------------------------------------------------------------------|
| MFCC           | Hệ số Cepstral tần số Mel — đặc trưng phổ của giọng nói.                                      |
| pYIN           | Thuật toán YIN xác suất để ước lượng tần số cơ bản (cao độ).                                   |
| IVFFlat        | Tệp đảo ngược với tính toán khoảng cách phẳng — loại chỉ mục ANN.                              |
| DTW            | Dynamic Time Warping — thuật toán căn chỉnh chuỗi, bền vững với biến thiên tốc độ.            |
| pgvector       | Tiện ích PostgreSQL để lưu trữ và tìm kiếm vector embedding.                                   |
| ANN            | Tìm kiếm láng giềng gần nhất xấp xỉ — nhanh nhưng không chính xác tuyệt đối.                   |
| Segment        | Khoảng âm thanh không im lặng được phát hiện bởi phát hiện im lặng.                            |
| minmax_vec     | Vector đặc trưng đã chuẩn hóa Min-Max dùng cho lập chỉ mục ANN.                                |
| raw_vec        | Vector đặc trưng chưa chuẩn hóa gốc, lưu để tái chuẩn hóa sau này.                             |
| FastDTW        | Xấp xỉ thời gian tuyến tính của DTW sử dụng phân giải đa mức.                                  |

---

*Tài liệu được tạo: 2026-04-16*

*Hệ thống: Tìm kiếm Độ tương tự Giọng nam — Dự án Cơ sở dữ liệu Đa phương tiện (M-csdl-dpt)*

*Phiên bản: v2*

--- 
