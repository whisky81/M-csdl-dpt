# Xây dựng hệ CSDL lưu trữ và tìm kiếm giọng nói đàn ông.
// lưu ý hệ thống chỉ lưu file trong csdl chỉ bao gồm file audio của đàn ông 
## yêu cầu bài toán
1. Hãy xây dựng/sưu tầm một bộ dữ liệu gồm ít nhất 500 files âm thanh về giọng nói đàn ông, các file có cùng độ dài (SV tùy chọn định dạng file âm thanh và độ dài của các files âm thanh).
2. Hãy xây dựng một bộ thuộc tính để nhận diện giọng nói đàn ông của các file âm thanh khác nhau từ bộ dữ liệu đã thu thập (bao gồm các thuộc tính thể hiện sự tương đồng giữa các giọng nói và sự khác biệt giữa các giọng nói). Trình bày cụ thể về lý do lựa chọn và các giá trị thông tin của các thuộc tính này.
3. Xây dựng hệ CSDL để lưu trữ các siêu dữ liệu và hỗ trợ tìm kiểm âm thanh dựa trên các siêu dữ liệu này.
4. Xây dựng hệ thống tìm kiếm âm thanh đàn ông với đầu vào là một file âm thanh mới của một người đàn ông nào đó (đối tượng đã có và không có trong dữ liệu), đầu ra là 5 files âm thanh giống nhất, xếp thứ tự giảm dần về độ tương đồng giọng nói với âm thanh đầu vào.
    - Trình bày sơ đồ khối của hệ thống và quy trình thực hiện yêu cầu của đề bài.
    - Trình bày các kết quả trung gian của quá trình tìm kiếm âm thanh trong hệ thống.
5. Demo hệ thống và đánh giá kết quả đã đạt được.


## notes, thông tin về demo hiện tại và yêu cầu  
1. số lượng file âm thanh trong cơ sở dữ liệu: 1329
2. thời lượng của các file nằm trong khoảng [1.5, 32.65] (đơn vị giây)
    -> nó không match với yêu cầu 500 file âm thanh có cùng độ dài
    -> nhưng nếu tính file có độ dài từ [4, 7] có 430 file (có thể xem như là gần yêu cầu)
3. trong demo vẫn chưa hiểu về thuộc tính nào thể hiện sự tương đồng và thuộc tính nào thể hiện sự khác biệt, và vẫn chưa nêu ra lý
    do lựa chọn và các giá trị thông tin -> viết 1 file markdown để làm rõ hơn về điều này
4. đã có hệ csdl và có 1 bảng 'voices_metadata' nhưng vẫn chưa hỗ trợ tìm kiếm âm thanh dựa trên các siêu dữ liệu này,
5. hãy viết thêm vào file markdown trên sơ đồ khối của hệ thống và quy trình thực hiện yêu cầu của đề bài, và các kết quả trung gian 

- thông tin về file metadata.csv bao gồm các cột
    ```
    audio_file,speaker_id,speaker_name,speaker_gender,chapter_id,chapter_title,book_id,book_title,utterance_id,chapter_duration_minutes,duration_seconds,duration_minutes,duration_source,duration_diff_percent,transcript_word_count,transcript_word_count_actual,speaking_rate_words_per_sec,file_size_bytes,subset,transcript_sample,transcript_full
    ```
- demo hiện tại có tạo ra list[vector(18)] (danh sách vector 18 chiều) -> tôi nghĩ nó chưa đủ vậy có bổ sung thêm gì nữa không

