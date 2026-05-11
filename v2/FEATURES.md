# Phân tích 18 chiều đặc trưng giọng nói nam

## Tổng quan

Hệ thống sử dụng **18 chiều đặc trưng** để biểu diễn mỗi phân đoạn (segment) âm thanh, bao gồm:

| Nhóm | Số chiều | Ký hiệu | Loại |
|:---|:---:|:---|:---|
| Thủ công (hand-crafted) | 5 | E, Z, S, P, C | Miền thời gian + tần số |
| MFCC (Mel-Frequency Cepstral Coefficients) | 13 | M₁ … M₁₃ | Miền tần số (thang Mel) |
| **Tổng** | **18** | | |

---

## 1. Năm đặc trưng thủ công (Hand-crafted Features)

### 1.1. Năng lượng trung bình — `avg_energy` (E)

| Thuộc tính | Giá trị |
|:---|:---|
| **Công thức** | E = mean(RMS) — Root Mean Square trên tất cả frame |
| **Đơn vị** | Biên độ chuẩn hóa (không thứ nguyên) |
| **Ý nghĩa** | Đo **độ to** trung bình của phân đoạn |

- **Thể hiện sự khác biệt**: Mỗi người nói có cường độ giọng khác nhau (người nói to, nhỏ), khoảng cách tới micro, và mức năng lượng tự nhiên khác nhau. Hai người đọc cùng một câu sẽ có `avg_energy` khác nhau.
- **Vai trò trong tìm kiếm**: Giúp phân biệt các bản ghi có điều kiện thu âm khác nhau. Một người nói nhỏ sẽ không bao giờ khớp với người nói to, ngay cả khi âm sắc tương tự.

---

### 1.2. Tốc độ cắt qua không trung bình — `avg_zcr` (Z)

| Thuộc tính | Giá trị |
|:---|:---|
| **Công thức** | ZCR = (số lần tín hiệu đổi dấu) / (độ dài frame) |
| **Đơn vị** | Lần/giây |
| **Ý nghĩa** | Ước lượng **tần số trung bình** của tín hiệu |

- **Thể hiện sự khác biệt**: ZCR phản ánh tỉ lệ giữa âm hữu thanh (nguyên âm — ZCR thấp) và âm vô thanh (phụ âm xát, tắc — ZCR cao). Người nói có cách phát âm khác nhau (nhấn mạnh phụ âm, kéo dài nguyên âm) sẽ có ZCR khác nhau.
- **Vai trò trong tìm kiếm**: Đây là đặc trưng phụ thuộc vào **nội dung ngữ âm**, không chỉ người nói. Cùng một người đọc hai câu khác nhau sẽ có ZCR khác nhau → góp phần vào độ phân biệt giữa các file.

---

### 1.3. Tỷ lệ im lặng — `silence_ratio` (S)

| Thuộc tính | Giá trị |
|:---|:---|
| **Công thức** | S = (số frame có dB < −40) / (tổng số frame) |
| **Đơn vị** | Tỷ lệ [0, 1] |
| **Ý nghĩa** | Đo **mức độ ngắt quãng** trong giọng nói |

- **Thể hiện sự khác biệt**: Người nói có phong cách khác nhau: người nói chậm, có nhiều khoảng lặng giữa các từ; người nói nhanh, liên tục, ít khoảng lặng. Đây là đặc trưng phong cách cá nhân.
- **Vai trò trong tìm kiếm**: Giúp phân biệt giọng nói (nhiều khoảng lặng) với nhạc (ít khoảng lặng), và phân biệt phong cách nói giữa các cá nhân.

---

### 1.4. Cao độ trung bình — `avg_pitch` (P)

| Thuộc tính | Giá trị |
|:---|:---|
| **Công thức** | F0 trung bình trên các frame hữu thanh (dùng PYIN algorithm) |
| **Dải tần** | 65.41 Hz (C2) – 2093 Hz (C7) |
| **Đơn vị** | Hz |
| **Ý nghĩa** | Tần số cơ bản — **độ cao** của giọng nói |

- **Thể hiện sự tương đồng**: Đây là đặc trưng **sinh lý** của người nói. Mỗi người có dải tần cơ bản đặc trưng do cấu trúc thanh quản (dây thanh dài/ngắn, dày/mỏng). Cùng một người, dù nói nội dung gì, F0 trung bình luôn nằm trong một khoảng hẹp.
- **Vai trò trong tìm kiếm**: Đặc trưng **chính** để xác định danh tính người nói. Hai file của cùng một người sẽ có `avg_pitch` rất gần nhau. Đây là một trong những đặc trưng quan trọng nhất cho **speaker identification**.

---

### 1.5. Tâm phổ trung bình — `avg_centroid` (C)

| Thuộc tính | Giá trị |
|:---|:---|
| **Công thức** | Spectral Centroid = Σ(fᵢ × Aᵢ) / Σ(Aᵢ), trung bình trên tất cả frame |
| **Đơn vị** | Hz |
| **Ý nghĩa** | Điểm "cân bằng" của phổ — đo **độ sáng** (brightness) của âm thanh |

- **Thể hiện sự tương đồng**: Cùng một người nói, cấu trúc formant (dải cộng hưởng của đường dẫn âm) ổn định → tâm phổ sẽ nằm trong khoảng nhất quán. Giọng nam trầm có centroid thấp hơn giọng nữ.
- **Thể hiện sự khác biệt**: Giữa những người khác nhau, tâm phổ thay đổi do khác biệt về hình dạng đường dẫn âm (họng, miệng, mũi).
- **Vai trò trong tìm kiếm**: Kết hợp với pitch, centroid giúp khoanh vùng người nói theo đặc tính vật lý.

---

## 2. Mười ba hệ số MFCC (Mel-Frequency Cepstral Coefficients)

### 2.1. Nguồn gốc và cách tính

| Bước | Mô tả |
|:---|:---|
| 1. **Pre-emphasis** | Khuếch đại tần số cao để cân bằng phổ |
| 2. **Framing + Windowing** | Chia tín hiệu thành các frame chồng lấn, nhân với cửa sổ Hamming |
| 3. **FFT** | Biến đổi Fourier để có phổ tần số |
| 4. **Mel Filterbank** | Áp dụng 26 bộ lọc tam giác trên thang Mel (mô phỏng tai người) |
| 5. **Log** | Lấy log năng lượng mỗi bộ lọc |
| 6. **DCT** | Biến đổi Cosine rời rạc (DCT-II) → 13 hệ số MFCC |

### 2.2. Ý nghĩa từng hệ số

| Hệ số | Tên gọi | Ý nghĩa |
|:---|:---|:---|
| **M₁** | MFCC-1 | Năng lượng tổng thể (liên quan đến độ to) |
| **M₂** | MFCC-2 | Cân bằng phổ — nghiêng về tần số thấp hay cao |
| **M₃** | MFCC-3 | Độ rộng phổ — formant thứ nhất (F1) |
| **M₄–M₆** | MFCC-4→6 | Formant bậc cao (F2, F3) — đặc trưng âm sắc |
| **M₇–M₁₃** | MFCC-7→13 | Chi tiết phổ tinh — cấu trúc hài âm và tạp âm |

### 2.3. Vai trò trong nhận dạng giọng nói

- **Thể hiện sự tương đồng**: Bộ MFCC được thiết kế để **bỏ qua kích thích nguồn (F0)** và chỉ giữ lại **đặc trưng bộ lọc đường dẫn âm** (vocal tract). Vì cấu trúc đường dẫn âm là **duy nhất cho mỗi người** (giống như vân tay), MFCC là đặc trưng chính cho speaker identification:
  - M₁–M₃ mô tả hình dạng tổng quát của phổ — khác biệt giữa các giọng
  - M₄–M₁₃ mô tả chi tiết tinh — đặc trưng riêng của từng cá nhân

- **Thể hiện sự khác biệt**: Trong cùng một người, MFCC thay đổi theo âm vị được phát âm. Hai từ khác nhau → chuỗi MFCC khác nhau → cần DTW để co giãn thời gian và so khớp.

---

## 3. Ma trận phân loại: Tương đồng vs Khác biệt

| # | Đặc trưng | Tương đồng<br>(cùng người) | Khác biệt<br>(khác người) | Giải thích |
|:---:|:---|:---:|:---:|:---|
| 1 | `avg_energy` | ★★ | ★★★ | Phụ thuộc vào cường độ giọng và điều kiện thu |
| 2 | `avg_zcr` | ★ | ★★★ | Phụ thuộc vào nội dung ngữ âm và cách phát âm |
| 3 | `silence_ratio` | ★★ | ★★★ | Phụ thuộc vào phong cách nói (ngắt nghỉ) |
| 4 | **`avg_pitch`** | ★★★★★ | ★★★★★ | **Đặc trưng sinh lý cốt lõi** — F0 của cùng người rất ổn định |
| 5 | **`avg_centroid`** | ★★★★ | ★★★ | Liên quan đến âm sắc, ổn định theo người |
| 6–8 | **MFCC 1–3** | ★★★ | ★★★★ | Hình dạng phổ tổng quát — khác biệt rõ giữa các giọng |
| 9–13 | **MFCC 4–8** | ★★★★ | ★★★ | Formant và cấu trúc âm sắc — đặc trưng cá nhân |
| 14–18 | **MFCC 9–13** | ★★★ | ★★★ | Chi tiết tinh — hữu ích cho DTW so khớp chuỗi |

### Tổng kết

- **Đặc trưng tương đồng (Speaker Identity)**: `avg_pitch` (P), `avg_centroid` (C), MFCC 4–8 (M₄–M₈) — đây là các đặc trưng ổn định cho cùng một người nói, bất kể họ đang nói nội dung gì.
- **Đặc trưng khác biệt (Utterance/Style)**: `avg_energy` (E), `avg_zcr` (Z), `silence_ratio` (S) — đây là các đặc trưng thay đổi theo nội dung, cảm xúc, và điều kiện ghi âm.
- **Đặc trưng hỗn hợp**: MFCC 1–3 và MFCC 9–13 — vừa phản ánh đặc tính người nói, vừa thay đổi theo nội dung âm vị. Chính sự kết hợp này khiến DTW trên chuỗi MFCC trở thành phương pháp hiệu quả để so khớp giọng nói.

---

## 4. Lý do lựa chọn bộ 18 đặc trưng

1. **Đại diện cho cả hai miền**: Miền thời gian (energy, ZCR, silence) và miền tần số (pitch, centroid, MFCC) — đảm bảo nắm bắt toàn diện đặc tính giọng nói.

2. **Tận dụng nghiên cứu về sinh lý giọng nói**: F0 và formant là hai yếu tố sinh lý quyết định sự khác biệt giọng nói giữa các cá nhân:
   - F0: do dây thanh quyết định (độ dài, độ căng, khối lượng)
   - Formant (thể hiện qua MFCC): do hình dạng đường dẫn âm quyết định

3. **Mô phỏng cơ chế nghe của tai người** (thang Mel): Thang Mel được xây dựng dựa trên thực nghiệm tâm lý-âm học — tai người phân biệt tần số tốt hơn ở dải thấp. MFCC nắm bắt thông tin theo cách tương tự tai người.

4. **Hiệu quả tính toán**: 13 MFCC là con số tiêu chuẩn trong xử lý tiếng nói, đủ để giữ lại thông tin quan trọng về đường dẫn âm nhưng đủ nhỏ để tính toán nhanh.

5. **Phù hợp với DTW**: Chuỗi vector 18 chiều cho mỗi segment cho phép tính khoảng cách Euclidean cục bộ trong DTW — các đặc trưng có thang đo tương đối đồng nhất sau khi chuẩn hóa Min-Max.

---

## 5. Giá trị thông tin của các thuộc tính

| Mức độ | Đặc trưng | Giá trị thông tin |
|:---|:---|:---|
| **Rất cao** | `avg_pitch`, MFCC 4–8 | Xác định danh tính người nói — "vân tay giọng nói" |
| **Cao** | `avg_centroid`, MFCC 1–3 | Phân biệt âm sắc giữa các giọng |
| **Trung bình** | `avg_zcr`, `silence_ratio`, MFCC 9–13 | Phân biệt phong cách nói và nội dung |
| **Thấp** | `avg_energy` | Phụ thuộc vào điều kiện ghi âm, hữu ích nhưng không ổn định |

> **Lưu ý**: Giá trị thông tin ở trên là trong ngữ cảnh **speaker identification**. Trong ngữ cảnh **speech recognition** (nhận dạng nội dung), thứ tự ưu tiên sẽ khác — MFCC 1–13 sẽ có giá trị cao nhất.

---

## 6. Quy trình sử dụng đặc trưng trong tìm kiếm

```
File âm thanh đầu vào (.flac)
│
├─ Phân đoạn (split by silence, top_db=40)
│   → Các segment [s₀, s₁, ..., sₙ]
│
├─ Trích xuất vector 18D cho mỗi segment
│   → [E, Z, S, P, C, M₁...M₁₃]
│
├─ Chuẩn hóa Min-Max (dùng global min/max từ toàn bộ CSDL)
│   → Đưa tất cả chiều về [0, 1]
│
├─ Tìm K segment gần nhất (IVFFlat hoặc KD-Tree)
│   → Khoảng cách Euclidean trên vector 18D
│
├─ Gom candidate files (theo tần suất xuất hiện)
│   → Top 50 file có nhiều segment khớp nhất
│
├─ DTW trên chuỗi segment của từng candidate
│   → So khớp toàn bộ chuỗi, không chỉ từng segment đơn lẻ
│
└─ Xếp hạng theo DTW distance tăng dần → Top 5
```
