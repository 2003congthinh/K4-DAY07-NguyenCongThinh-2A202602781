# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Công Thịnh
**Nhóm:** G69
**Ngày:** 20/9/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao nghĩa là hai vector đang chỉ theo cùng một hướng trong không gian embedding, tức là chúng có cùng ý nghĩa hoặc cùng chủ đề dù có thể dùng từ khác nhau. Khi cosine gần 1, hai đoạn văn bản được xem là tương đồng cao.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Hàng hóa giao đến bị hỏng và tôi muốn đổi trả." 
- Câu B: "Sản phẩm nhận được không đúng tình trạng, tôi cần hoàn trả." 
- Tại sao tương đồng: cả hai câu nói về cùng một tình huống và cùng mục tiêu là yêu cầu đổi trả do sản phẩm hỏng. Từ vựng khác nhau nhưng ý nghĩa gần như nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Tôi muốn đổi trả do sản phẩm bị lỗi." 
- Câu B: "Mặt trời mọc ở phía đông mỗi sáng." 
- Tại sao khác: nghĩa là hoàn toàn khác nhau, dù cùng có từ "sản phẩm" ở một câu nhưng nội dung không liên quan.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine đo hướng của vector, tức là đo ngữ nghĩa, trong khi Euclidean đo khoảng cách tuyệt đối trong không gian. Với embeddings văn bản, khoảng cách Euclid dễ bị ảnh hưởng bởi độ dài vector chứ không phản ánh đúng sự tương đồng về ý nghĩa; cosine phù hợp hơn khi so sánh các vector đã được chuẩn hóa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức: $\lceil (L - overlap) / (chunk\_size - overlap) \rceil$ 
>
> $\lceil (10000 - 50) / (500 - 50) \rceil = \lceil 9950 / 450 \rceil = \lceil 22.11 \rceil = 23$ chunk.
>
> **Đáp án:** 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Nếu overlap = 100: $\lceil (10000 - 100) / (500 - 100) \rceil = \lceil 9900 / 400 \rceil = \lceil 24.75 \rceil = 25$ chunk. Vậy số chunk tăng lên 25. Overlap lớn giúp giữ ngữ cảnh giữa các chunk liền kề, nhưng nó cũng làm tăng số lượng chunk và chi phí xử lý, nên cần lựa chọn cân bằng giữa độ liên tục và hiệu quả.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi chia văn bản thành các câu bằng regex `(?<=[.!?])\s+`, nghĩa là tách ở vị trí sau dấu kết thúc câu và giữ lại dấu câu. Sau đó, tôi gom từng nhóm `max_sentences_per_chunk` câu lại thành một chunk, rồi dùng `strip()` để loại bỏ khoảng trắng thừa. Edge case cần lưu ý là các trường hợp như chữ viết tắt, số thập phân, hoặc câu ngắn có thể bị cắt sai nếu dùng regex quá đơn giản.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán làm việc theo kiểu đệ quy: bắt đầu từ separator ưu tiên cao nhất, cắt văn bản theo ranh giới ngữ nghĩa trước, nếu một mảnh vẫn dài hơn `chunk_size` thì gọi lại `_split` với separator tiếp theo. Base case là khi `len(current_text) <= chunk_size`, khi không còn separator để dùng, hoặc khi separator rỗng thì thực hiện cắt cứng theo kích thước. Cách này giúp giữ ngữ nghĩa và tránh tạo các mảnh quá ngắn, đồng thời vẫn đảm bảo văn bản luôn được tách đúng giới hạn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tôi lưu từng tài liệu dưới dạng record có `id`, `content`, `embedding` và `metadata`, rồi gắn thêm `doc_id` mặc định nếu chưa có. Khi search, tôi lấy embedding của câu hỏi và tính dot product với mỗi embedding đã lưu, sau đó sắp xếp giảm dần theo score để lấy top-k. Vì mock embedder đã chuẩn hóa vector, dot product tương đương với cosine similarity nên cách này phù hợp với yêu cầu bài lab.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Cách làm đúng là lọc metadata trước khi chạy similarity search, nhằm giảm không gian tìm kiếm và tránh lấy những chunk không phù hợp với đối tượng người dùng. Với `delete_document`, tôi xóa tất cả record có `metadata['doc_id'] == doc_id`, và trả về `True` nếu xóa thành công, `False` nếu không có doc nào khớp.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Tôi lấy top-k chunks từ `EmbeddingStore.search()`, ghép các đoạn này thành một prompt có tiền đề "Relevant context" và đưa vào `llm_fn` cùng câu hỏi của người dùng. Cách này giữ nguyên ngữ cảnh cần thiết, tránh prompt thiếu thông tin, và giúp model trả lời dựa trên dữ liệu đã truy xuất thay vì đoán mò.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```bash
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-7.4.0, pluggy-1.0.0 -- C:\Users\Avita V14\anaconda3\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\Avita V14\Desktop\code_folders\Vin AI20K\Tut\K4-DAY07-NguyenCongThinh-2A202602781
plugins: anyio-4.2.0
collected 42 items

... (các test tương ứng với chunking, store, agent, similarity)

============================= 42 passed in 0.32s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Hàng hóa giao đến bị hỏng và tôi muốn đổi trả. | Sản phẩm nhận được không đúng tình trạng, tôi cần hoàn trả. | cao | 0.1159 | Có |
| 2 | Tôi muốn đổi trả do sản phẩm bị lỗi. | Mặt trời mọc ở phía đông mỗi sáng. | thấp | -0.0494 | Có |
| 3 | Tôi muốn biết thời hạn bảo hành của sản phẩm. | Sản phẩm này được bảo hành trong bao lâu? | cao | 0.0598 | Có |
| 4 | Nhà Bán cần lưu video đóng gói hàng hóa. | Khách hàng muốn theo dõi tình trạng giao hàng. | thấp | 0.0815 | Không |
| 5 | Tôi cần đổi sản phẩm vì nhận sai màu. | Tôi muốn hoàn tiền vì sản phẩm bị giao nhầm màu. | cao | -0.0097 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là cặp 5: hai câu cùng nói về sản phẩm bị giao sai màu nhưng điểm cosine chỉ là -0.0097, thấp hơn nhiều so với cặp 4 có nội dung ít tương đồng hơn nhưng đạt 0.0815. Điều này cho thấy `MockEmbedder` hiện tại không biểu diễn đáng tin cậy quan hệ ngữ nghĩa; điểm số chủ yếu phản ánh vector giả lập được tạo từ chuỗi đầu vào. Vì vậy, cần dùng embedding đa ngữ thật như `LocalEmbedder` để đánh giá semantic similarity công bằng hơn.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời gian hỗ trợ đổi trả hàng tại Tiki là bao nhiêu ngày? (`buyer`) | `buyer-chinh-sach-bao-hanh`: "Tiki bảo hành bằng các hình thức nào? ..."; đây là nội dung bảo hành, không phải mục thời hạn đổi trả. | 0.2455 | Không ở top-1 và không thấy chunk gold trong top-3 | `bench.py` không gọi `KnowledgeBaseAgent`, nên chưa có agent answer được sinh ra. Nếu trả lời theo gold answer thì thời hạn là 30 ngày kể từ khi nhận hàng thành công, nhưng kết quả top-3 chưa cung cấp đúng chunk này. |
| 2 | Sản phẩm bảo hành gửi về Tiki thì mất bao lâu để nhận lại? (`buyer`) | `buyer-chinh-sach-doi-tra-hoan-tien`: "Chính sách đổi trả – Các thông tin cần thiết khi yêu cầu đổi trả..."; không phải mục thời gian nhận lại sản phẩm bảo hành. | 0.3730 | Không ở top-1; có chunk liên quan trong top-3 ở hạng 2 | `bench.py` không sinh câu trả lời của agent. Gold answer là khoảng 15-30 ngày tùy linh kiện, không tính thời gian vận chuyển; sản phẩm Apple khoảng 30-60 ngày. |
| 3 | Nhà Bán cần lưu trữ video đóng gói hàng hóa trong bao lâu? (`seller`) | `seller-faq-doi-tra-bao-hanh`: "Đối với hàng hóa có hạn sử dụng... Khách hàng được phép yêu cầu đổi – trả..."; không phải mục thời hạn lưu video. | 0.2412 | Không ở top-1 và không thấy chunk gold trong top-3 | `bench.py` không sinh câu trả lời của agent. Gold answer là Nhà Bán phải lưu video đóng gói tối thiểu 45 ngày; kết quả top-3 chưa truy xuất được chunk chứa con số này. |
| 4 | Những nhóm sản phẩm nào không được đổi trả theo nhu cầu? (`buyer`) | `buyer-chinh-sach-doi-tra-hoan-tien`: "Chính sách đổi trả – Các thông tin cần thiết khi yêu cầu đổi trả..."; chưa phải danh sách nhóm sản phẩm hạn chế. | 0.3109 | Không ở top-1; top-3 có cùng tài liệu hạn chế ở hạng 3 nhưng snippet chưa chứa gold answer | `bench.py` không sinh câu trả lời của agent. Gold answer gồm các nhóm như đồ lót/đồ bơi, nước hoa, trang sức, báo/tạp chí, hoa tươi/cây cảnh và thực phẩm tươi sống; kết quả top-3 chưa cho thấy chunk danh sách đầy đủ. |
| 5 | Quy trình xử lý bảo hành của Nhà Bán gồm những bước nào? (`seller`) | `seller-fbt-quy-trinh-doi-tra-bao-hanh`: "Nguyên tắc chung - Đối với hàng hóa đủ điều kiện đổi – trả: Tiki lên đơn hàng mới..."; liên quan nhưng chưa nêu đầy đủ các bước bảo hành. | 0.2109 | Không ở top-1; có chunk liên quan trong top-3 ở hạng 3 | `bench.py` không sinh câu trả lời của agent. Top-3 có chunk bắt đầu bằng "Quy trình xử lý bảo hành - Bước 1...", nên có thể dùng làm ngữ cảnh để trả lời các bước tiếp theo. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 2 / 5 (câu 2 và câu 5)

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua kết quả benchmark, tôi nhận ra rằng metadata filter giúp giới hạn đúng nhóm `buyer` hoặc `seller`, nhưng chưa đủ để chọn đúng section trong cùng một tài liệu. Với `MockEmbedder`, điểm score cao nhất đôi khi vẫn thuộc về chunk chỉ liên quan về chủ đề; vì vậy cần đánh giá cả nội dung chunk và vị trí của chunk trong top-3, không chỉ nhìn vào score.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5/ 5 |
| Hướng tiếp cận của tôi (My Approach) | 10/ 10 |
| Hoàn thiện code (Core Implementation — tests) | 30/ 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 3/ 5 |
| Kết quả truy xuất của tôi (Competition Results) | 2/ 10 |
| **Tổng phần cá nhân** | **50/ 60** |
