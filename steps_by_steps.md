# HƯỚNG 
Dưới đây là hướng dẫn siêu chi tiết để bạn xây dựng hệ thống CSDL lưu trữ và tìm kiếm giọng nói đàn ông, dựa trên các nguyên lý kỹ thuật từ các nguồn tài liệu chuyên ngành:

### 1. Xây dựng bộ dữ liệu (Dataset)
Để đảm bảo tính nhất quán DẪNcho việc trích rút đặc trưng sau này, bạn cần thực hiện các bước chuẩn bị dữ liệu thô:
*   **Số lượng:** Ít nhất 500 file âm thanh giọng nam.
*   **Định dạng khuyến nghị:** Nên chọn **.WAV** (dạng chưa nén) để giữ nguyên chất lượng tín hiệu cho việc phân tích đặc trưng chính xác nhất, hoặc **.MP3** (MPEG-Audio Layer 3) để tiết kiệm dung lượng nếu cần.
*   **Thông số kỹ thuật đồng nhất:** 
    *   **Độ dài:** Tất cả các file phải có cùng thời lượng (ví dụ: 5 giây hoặc 10 giây).
    *   **Tần số lấy mẫu (Sampling rate):** Nên để **8.000 Hz** (tiêu chuẩn điện thoại cho tiếng người) hoặc **16.000 Hz** để đảm bảo phủ hết dải tần tiếng nói (thường dưới 7.000 Hz).
    *   **Độ sâu bit (Bit depth):** 8-bit hoặc 16-bit.
*   **Nguồn sưu tầm:** Có thể thu âm trực tiếp hoặc trích xuất từ các đoạn hội thoại, bài diễn thuyết, hoặc các bộ dữ liệu mở về tiếng nói.

### 2. Xây dựng bộ thuộc tính nhận diện (Feature Engineering)
Dựa trên đặc tính âm học của giọng nam, bạn cần trích rút các **đặc trưng bậc thấp (low-level features)** sau để tạo thành một **vectơ đặc trưng** cho mỗi file:

#### A. Thuộc tính miền thời gian (Time-domain)
*   **Năng lượng trung bình (Average Energy):** Đo độ lớn (loudness) của giọng nói. Giúp phân biệt giữa giọng nói to, rõ ràng với tiếng thì thầm.
*   **Tốc độ đổi dấu (Zero-Crossing Rate - ZCR):** Giọng nam có tính biến thiên ZCR rất cao, đặc biệt ở các phụ âm ma sát. Đây là thuộc tính quan trọng để phân biệt tiếng người với âm nhạc.
*   **Tỷ lệ khoảng lặng (Silence Ratio):** Giúp nhận diện nhịp điệu nói (người nói nhanh hay chậm, ngắt nghỉ nhiều hay ít).

#### B. Thuộc tính miền tần số (Frequency-domain)
*   **Độ cao âm (Pitch):** Đây là thuộc tính **quan trọng nhất** để nhận diện giọng nam. Giọng nam trưởng thường có tần số cơ bản (Fundamental Frequency) thấp hơn nhiều so với giọng nữ hoặc trẻ em.
*   **Trọng tâm phổ (Spectral Centroid/Brightness):** Giọng nam thường có "độ sáng" thấp do năng lượng tập trung chủ yếu ở các dải tần thấp.
*   **Hệ số Mel-frequency cepstral (MFCCs):** Đây là đặc trưng "vàng" trong nhận dạng tiếng nói (ASR) vì nó mô phỏng cách tai người cảm nhận âm thanh, giúp phân biệt đặc điểm cá nhân của từng người nói.

**Lý do lựa chọn:** Sự kết hợp giữa Pitch (nhận diện giới tính/độ trầm) và MFCCs (nhận diện cá nhân) giúp hệ thống không chỉ biết đó là giọng nam mà còn biết đó là của người đàn ông nào.

### 3. Xây dựng hệ CSDL lưu trữ siêu dữ liệu
Bạn nên sử dụng mô hình **Quan hệ - Đối tượng (Object-Relational)** để quản lý:
*   **Cấu trúc lưu trữ:**
    *   **Dữ liệu thô:** Lưu trữ các file âm thanh gốc trong hệ thống tệp hoặc dưới dạng **BLOB** trong CSDL.
    *   **Siêu dữ liệu (Metadata):** Lưu vào bảng SQL gồm: ID, Tên người nói, Ngày thu âm, Định dạng, và quan trọng nhất là **Vectơ đặc trưng** (một mảng số thực đại diện cho các thuộc tính ở mục 2).
*   **Đánh chỉ số (Indexing):** Để tìm kiếm nhanh giữa 500 file, hãy sử dụng cấu trúc **Cây k-d (k-dimensional tree)** hoặc **Cây R (R-tree)** để tổ chức các vectơ đặc trưng trong không gian đa chiều.

### 4. Hệ thống tìm kiếm âm thanh (Similarity Search)
#### a. Sơ đồ khối và Quy trình thực hiện:
1.  **Đầu vào:** Một file âm thanh mới (Query).
2.  **Số hóa & Tiền xử lý:** Chuẩn hóa file query về cùng định dạng/tần số lấy mẫu với kho dữ liệu.
3.  **Trích rút đặc trưng:** Tính toán vectơ đặc trưng của file query (giống bước 2).
4.  **Tính toán độ tương đồng (Similarity Computation):** So sánh vectơ query với các vectơ đã lưu trong CSDL bằng **khoảng cách Euclidean (L2-norm)** hoặc **Cosines**. Khoảng cách càng nhỏ, độ tương đồng càng cao.
5.  **Sắp xếp & Kết quả:** Sắp xếp các kết quả theo thứ tự tăng dần của khoảng cách (giảm dần của độ tương đồng) và lấy 5 kết quả đầu tiên.

#### b. Các kết quả trung gian:
*   Vectơ số thực đại diện cho file query (ví dụ: $[0.5, 1.2, -0.3, ...]$).
*   Giá trị khoảng cách (distance) giữa query và từng file trong kho (ví dụ: $d_1=0.02, d_2=0.15,...$).
*   Danh sách ID của 5 file có khoảng cách nhỏ nhất.

### 5. Đánh giá kết quả
Để biết hệ thống hoạt động tốt đến đâu, bạn cần sử dụng hai thước đo:
*   **Độ chính xác (Precision):** Trong 5 kết quả trả về, có bao nhiêu file thực sự là của người đàn ông đó (nếu đối tượng đã có trong kho) hoặc có chất giọng giống nhất.
*   **Độ phủ (Recall):** Hệ thống có tìm sót những file giống người đó trong kho hay không.
*   **Phản hồi liên quan (Relevance Feedback):** Cho phép người dùng đánh giá kết quả để hệ thống tự điều chỉnh trọng số tìm kiếm cho lần sau.

---
# ĐÁNH GIÁ + SƠ ĐỒ KHỐI + QUY TRÌNH
Dựa trên các tài liệu đã cung cấp và bối cảnh môn học Hệ CSDL đa phương tiện, dưới đây là tư vấn và trình bày chi tiết cho yêu cầu của bạn:

### 1. Đánh giá sự hợp lý của công nghệ lựa chọn
Việc lựa chọn **Python**, **Postgres**, **HTML** và **Flask** để làm demo cho hệ thống tìm kiếm giọng nói là **hoàn toàn hợp lý** và phù hợp với các nguyên tắc kiến trúc được mô tả trong các nguồn tài liệu:

*   **Python:** Rất mạnh trong việc xử lý tín hiệu và trích rút đặc trưng (Feature Extraction). Các thư viện Python giúp thực hiện các phép biến đổi Fourier rời rạc (DFT) để trích xuất các thuộc tính bậc thấp như Pitch, MFCCs hay Spectral Centroid một cách tự động.
*   **PostgreSQL:** Phù hợp với mô hình **CSDL Quan hệ - Đối tượng**. Postgres có thể lưu trữ các thuộc tính cấu trúc (tên người nói, ID) và siêu dữ liệu (metadata) dưới dạng bảng. Đặc biệt, Postgres hỗ trợ lưu trữ mảng (array) rất tốt cho các vectơ đặc trưng âm thanh.
*   **Flask & HTML:** Đóng vai trò là **Giao diện người dùng (User Interface)** trong kiến trúc MIRS (Multimedia Information Retrieval System). Flask là một Framework nhẹ giúp xây dựng các chức năng truy vấn, duyệt (browsing) và hiển thị kết quả dưới dạng danh sách xếp hạng theo độ tương đồng.

---

### 2. Sơ đồ khối hệ thống và Quy trình thực hiện

Dựa trên mô hình tìm kiếm thông tin tổng quát và kiến trúc MIRS trong tài liệu, sơ đồ khối và quy trình được thiết kế như sau:

#### a. Sơ đồ khối hệ thống (System Architecture)
Hệ thống được chia thành hai pha chính: **Pha lưu trữ (Insertion)** và **Pha truy vấn (Retrieval)**.

1.  **Khối Giao diện (User Interface):** Tiếp nhận file âm thanh đầu vào (Query) và hiển thị kết quả.
2.  **Khối Trích rút đặc trưng (Feature Extractor):** Số hóa tín hiệu và chuyển đổi file âm thanh thành các vectơ đặc trưng (Vectơ hóa) bằng các phép biến đổi miền thời gian và tần số.
3.  **Khối Quản trị CSDL (MM-Database):** Lưu trữ file âm thanh gốc (dưới dạng đường dẫn hoặc BLOB) và các vectơ đặc trưng tương ứng trong Postgres.
4.  **Khối Tìm kiếm tương đồng (Similarity Search Engine):** Thực hiện tính toán khoảng cách (như khoảng cách Euclidean/L2-norm) giữa vectơ của câu truy vấn và các vectơ có sẵn trong kho dữ liệu.
5.  **Khối Sắp xếp và Kết quả (Result Presentation):** Lọc ra 5 kết quả có khoảng cách nhỏ nhất (độ tương đồng cao nhất) và trả về giao diện.

#### b. Quy trình thực hiện chi tiết

**Bước 1: Xây dựng bộ dữ liệu (Offline Phase)**
*   Thu thập 500 file giọng nam, thực hiện số hóa với tần số lấy mẫu (Sampling rate) và độ sâu bit (Bit depth) đồng nhất.
*   Dùng bộ trích rút đặc trưng để tính toán vectơ thuộc tính cho từng file (Pitch, Năng lượng, ZCR...).
*   Lưu các thông tin này vào PostgreSQL.

**Bước 2: Xử lý truy vấn (Online Phase)**
*   Người dùng tải lên một file âm thanh mới qua Flask UI.
*   Hệ thống tự động trích rút vectơ đặc trưng của file này theo cùng phương pháp ở Bước 1.

**Bước 3: So sánh tương đồng và Xếp hạng**
*   Hệ thống lấy vectơ truy vấn so sánh với 500 vectơ trong Postgres.
*   Sử dụng công thức tính khoảng cách (L2-norm) để ño sự khác biệt.
*   Sắp xếp toàn bộ dữ liệu theo thứ tự độ tương đồng giảm dần.

**Bước 4: Trả kết quả và Đánh giá**
*   Hiển thị 5 file âm thanh có độ tương đồng cao nhất lên màn hình HTML.
*   Đánh giá độ chính xác của hệ thống thông qua các chỉ số **Precision** (độ chính xác) và **Recall** (độ phủ) dựa trên phản hồi của người dùng.

---
# STEPS BY STEPS 
Dựa trên sơ đồ khối và các yêu cầu kỹ thuật đã thảo luận, dưới đây là kế hoạch chi tiết từng bước (Step-by-step Plan) để bạn hoàn thành hệ thống tìm kiếm giọng nói đàn ông sử dụng **Python, PostgreSQL và Flask**:

### Giai đoạn 1: Chuẩn bị và Thu thập dữ liệu (Dữ liệu thô)
*   **Bước 1: Xác định thông số kỹ thuật đồng nhất.** Thiết lập tiêu chuẩn cho 500 file: Định dạng `.wav` (không nén để giữ chất lượng), tần số lấy mẫu ($f_s$) 16.000 Hz (đảm bảo theo định lý Nyquist $f_s \ge 2f_{max}$ với tiếng người), độ sâu bit 16-bit, và độ dài cố định 5 giây,.
*   **Bước 2: Thu thập 500+ file giọng nam.** Bạn có thể thu âm hoặc sử dụng các tập dữ liệu mở (như LibriSpeech). Lưu trữ các file này trong một thư mục cố định (ví dụ: `/static/audio_db/`).
*   **Bước 3: Gán nhãn siêu dữ liệu cơ bản.** Tạo một file danh sách (CSV hoặc Excel) ghi lại ID, tên người nói (nếu có), và đường dẫn tệp để chuẩn bị đưa vào CSDL,.

### Giai đoạn 2: Xây dựng bộ trích rút đặc trưng (Feature Extractor)
*   **Bước 4: Cài đặt thư viện xử lý.** Sử dụng Python với thư viện `Librosa` hoặc `SciPy` để đọc file âm thanh và thực hiện các phép biến đổi Fourier.
*   **Bước 5: Lập trình hàm trích rút đặc trưng bậc thấp.** Viết hàm Python để tính toán cho mỗi file một **Vectơ đặc trưng** gồm:
    *   **Miền thời gian:** Năng lượng trung bình ($E$), Tốc độ đổi dấu ($ZCR$).
    *   **Miền tần số:** Cao độ (Pitch), Trọng tâm phổ (Spectral Centroid) và đặc biệt là **MFCCs** (13 hoặc 20 hệ số) để nhận diện đặc điểm giọng nói cá nhân.
*   **Bước 6: Chuẩn hóa dữ liệu.** Áp dụng kỹ thuật chuẩn hóa để các giá trị thuộc tính nằm trong cùng một khoảng (0 đến 1), tránh việc một thuộc tính có giá trị lớn (như tần số) làm át đi các thuộc tính khác khi tính khoảng cách.

### Giai đoạn 3: Thiết lập CSDL PostgreSQL (Storage Layer)
*   **Bước 7: Thiết kế bảng dữ liệu.** Tạo bảng trong Postgres (ví dụ: `male_voices`) với các cột: `id (Serial)`, `file_path (Text)`, `speaker_name (Text)`, và `feature_vector (float8[])`.
*   **Bước 8: Đổ dữ liệu vào kho (Offline Phase).** Chạy một script Python quét 500 file -> Trích rút vectơ đặc trưng -> `INSERT` toàn bộ vào Postgres.
*   **Bước 9: Đánh chỉ số (Indexing).** Với 500 bản ghi, việc tìm kiếm tuyến tính là đủ nhanh. Tuy nhiên, để đúng lý thuyết môn học, bạn có thể tìm hiểu cấu trúc **Cây k-d** hoặc sử dụng extension `pgvector` của Postgres để tối ưu hóa truy vấn vectơ.

### Giai đoạn 4: Phát triển bộ máy tìm kiếm tương đồng (Retrieval Engine)
*   **Bước 10: Cài đặt công thức tính khoảng cách.** Sử dụng **Khoảng cách Euclidean (L2-norm)** hoặc độ tương đồng Cosine để so sánh hai vectơ.
*   **Bước 11: Lập trình hàm "Top 5".** Viết truy vấn lấy vectơ đặc trưng của file Query -> Tính khoảng cách với tất cả vectơ trong DB -> Sắp xếp tăng dần theo khoảng cách -> Lấy 5 kết quả đầu tiên.

### Giai đoạn 5: Xây dựng giao diện Web (Flask UI)
*   **Bước 12: Thiết kế giao diện HTML.** Tạo trang web đơn giản cho phép người dùng tải lên (Upload) một file âm thanh mới để tìm kiếm.
*   **Bước 13: Xử lý Backend (Flask).** Viết Route trong Flask để tiếp nhận file, lưu tạm thời, gọi bộ trích rút đặc trưng và bộ máy tìm kiếm để trả về kết quả.
*   **Bước 14: Hiển thị kết quả.** Trình bày 5 kết quả dưới dạng danh sách, bao gồm trình phát âm thanh (audio player) để người dùng có thể nghe và so sánh trực tiếp.

### Giai đoạn 6: Đánh giá và Demo
*   **Bước 15: Kiểm thử hệ thống.** Thử nghiệm với trường hợp người nói đã có trong DB và người nói hoàn toàn mới.
*   **Bước 16: Tính toán độ chính xác.** Dựa trên 5 kết quả trả về, tính toán chỉ số **Precision** (Độ chính xác) và **Recall** (Độ phủ) để đánh giá chất lượng hệ thống.
*   **Bước 17: Hoàn thiện báo cáo.** Trình bày sơ đồ khối, các kết quả trung gian (như giá trị vectơ của Query, giá trị khoảng cách tính được) và demo thực tế.
