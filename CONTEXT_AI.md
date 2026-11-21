# TÀI LIỆU TỔNG QUAN DỰ ÁN (PROJECT CONTEXT MASTER DOC)

**Nghiên cứu chiến lược suy luận thích ứng (Adaptive Inference) trong Masked Diffusion Models cho bài toán giải đố logic**

* **Môn học:** Introduction to AI | **Nhóm thực hiện:** Nhóm 5 (5 thành viên)
* **Bài báo gốc:** *"Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions"* (ICML 2025)

---

## 1. Mục đích của tài liệu này (Dành cho AI / Notebook Context)
Tài liệu này đóng vai trò là nguồn ngữ cảnh trung tâm (System Context / Source of Truth) lưu trữ trong Notebook hoặc Workspace. Khi mở một khung chat mới hoặc khởi chạy Coding Agent, trợ lý AI sẽ đọc tài liệu này để nắm bắt ngay lập tức đề tài, kiến trúc mô hình, thuật toán suy luận, dữ liệu, các thước đo đánh giá và kế hoạch triển khai của Nhóm 5 mà không cần người dùng mô tả lại từ đầu.

---

## 2. Bài toán nghiên cứu & Động lực lý thuyết
* **Bối cảnh mô hình:** Mô hình khuếch tán che token (Masked Diffusion Models - MDM) có khả năng giải mã token theo thứ tự tùy ý (order-agnostic), nhưng trong lúc huấn luyện phải học tổ hợp hàm mũ các bài toán điền chỗ trống (infilling), trong đó có nhiều bài toán con cực kỳ khó về mặt tính toán (*Train for the Worst*).
* **Phát hiện cốt lõi từ bài báo ICML 2025:** Ở giai đoạn suy luận (inference), việc giải mã ngẫu nhiên (Vanilla MDM) dễ rơi vào các bài toán con khó chưa được học tốt. Tuy nhiên, nếu dùng chiến lược suy luận thích ứng (Adaptive Inference) ưu tiên mở khóa các vị trí `[MASK]` dễ đoán nhất trước (*Plan for the Best*), mô hình sẽ tự động né được các bài toán con khó mà không cần huấn luyện lại.
* **Hiệu quả thực nghiệm trên Sudoku:** Trên bài toán Sudoku (nơi thứ tự suy luận phụ thuộc vào từng bảng cụ thể chứ không cố định từ trái sang phải), chiến lược *Top Probability Margin* giúp tăng độ chính xác của MDM $6\text{M}$ tham số từ $6.88\%$ (ngẫu nhiên) lên $89.49\%$, vượt qua cả mô hình tự hồi quy (ARM $42\text{M}$ tham số) được huấn luyện có giám sát theo thứ tự giải đúng ($87.18\%$).
* **Mục tiêu của Nhóm 5:** Kiểm chứng hiện tượng này ở quy mô nhỏ (trên Google Colab Free) với Sudoku $4\times4$, $6\times6$ và tập con $9\times9$; đồng thời đối sánh trực tiếp với mô hình Tự hồi quy (ARM) và các giải thuật tìm kiếm thỏa mãn ràng buộc (CSP) cổ điển trong môn Nhập môn AI.

---

## 3. Phương pháp tiếp cận & Các mô hình so sánh
Nhóm thực hiện huấn luyện và đối sánh trực tiếp giữa hai nhánh: Học sâu tạo sinh (MDM, ARM) và Giải thuật CSP cổ điển.

| Nhóm phương pháp | Tên Mô hình / Chiến lược | Chi tiết kỹ thuật & Quy tắc hoạt động |
| :--- | :--- | :--- |
| **Autoregressive (ARM)** | **ARM Trái-sang-phải** | Mô hình Transformer Causal (kiến trúc GPT-2 nhỏ $\sim 4\text{M}-6\text{M}$ tham số), sinh tuần tự cố định từ ô đầu đến ô cuối. |
| **Masked Diffusion (MDM)** | **MDM - Random (Vanilla)** | Mô hình MDM (cùng quy mô tham số $\sim 6\text{M}$), ở mỗi bước chọn mở khóa ngẫu nhiên các vị trí đang là token `[MASK]`. |
| **Masked Diffusion (MDM)** | **MDM - Top Probability** | Chọn mở khóa $K$ vị trí `[MASK]` có xác suất dự đoán cao nhất: $\max_j p_\theta(x^i = j \mid x_t)$. Dễ bị nhầm khi có 2 số cùng xác suất cao xấp xỉ nhau. |
| **Masked Diffusion (MDM)** | **MDM - Top Probability Margin** | Chọn mở khóa $K$ vị trí `[MASK]` có độ chênh lệch lớn nhất giữa xác suất cao nhất ($\text{top-1}$) và cao thứ hai ($\text{top-2}$): $\|p(j_1) - p(j_2)\|$. Đo độ chắc chắn chính xác nhất. |
| **Giải thuật AI cổ điển (CSP Baselines)** | **Backtracking, MRV + Forward Checking, AC-3** | Các thuật toán tìm kiếm và lan truyền ràng buộc chuẩn trong môn Introduction to AI. Đặc biệt, heuristic MRV (chọn biến ít giá trị hợp lệ còn lại nhất) có sự tương đồng trực tiếp với quy tắc Margin của MDM. |

---

## 4. Cấu hình kỹ thuật & Huấn luyện (Tối ưu cho Google Colab Free)
* **Kiến trúc mạng:** Transformer nhỏ (cấu hình tiny / GPT-2 khoảng $4\text{M} - 6\text{M}$ tham số: $4-6$ layers, $4-8$ heads, `hidden_size = 256`). Đối chiếu cách cài đặt với repo của Ye et al. (2024) và DreamLM/Dream.
* **Biểu diễn Input / Output:** Input là bảng Sudoku trải phẳng thành chuỗi 1D, trong đó các ô trống được gán token `[MASK]` (hoặc ID `0`). Output là bảng Sudoku đầy đủ hợp lệ kèm theo lịch sử thứ tự các ô được mở qua từng bước thời gian.
* **Quá trình huấn luyện MDM:** Ở mỗi batch, lấy tỷ lệ che ngẫu nhiên $t \in [0, 1]$, che ngẫu nhiên các ô thành `[MASK]` và tính hàm mất mát Cross-Entropy chỉ trên các vị trí bị che. Cấu hình tham khảo từ paper: `learning_rate = 0.001`, AdamW, `batch_size = 128-256`.
* **Quá trình suy luận MDM (Inference):** Sử dụng khoảng $50$ bước lấy mẫu ngược (reverse sampling steps) hoặc mở từng ô một cho bảng nhỏ. Trong paper gốc, hàm chọn vị trí $\mathcal{F}(\theta, x_t)$ được cộng thêm nhiễu Gumbel (hệ số $0.5$) để tối ưu hiệu quả.

---

## 5. Dữ liệu & Thước đo đánh giá (Evaluation Metrics)
* **Nguồn dữ liệu:** 
  1. Sudoku $4\times4$ và $6\times6$ do nhóm tự sinh bằng solver cổ điển (khoảng $10\text{k}$ mẫu cho giai đoạn Pilot).
  2. Bộ dữ liệu $3$ triệu câu đố Sudoku $9\times9$ của Radcliffe (2020) trên Kaggle, dự kiến trích xuất $100\text{k} - 200\text{k}$ mẫu để huấn luyện chính thức.
* **Phân chia độ khó:** Chia theo số lượng ô cho trước (givens) và đối chiếu với cách chia của paper (bảng giải được bằng $7$ chiến lược logic cố định so với bảng khó bắt buộc phải dùng backtracking).
* **Các chỉ số đánh giá chính:** 
  1. Độ chính xác giải đúng toàn bảng (Board-level accuracy).
  2. Độ chính xác từng ô (Cell-level accuracy).
  3. Số vi phạm ràng buộc Sudoku (Constraint violations).
  4. Số bước và thời gian giải.
* **Phương pháp kiểm định:** Chạy ít nhất $3$ lần ($3$ random seeds), báo cáo trung bình và độ lệch chuẩn; so sánh ghép cặp (paired comparison) trên cùng tập bảng test; phân tích sâu độ chính xác tại các bước có margin cao so với bước có margin thấp.

---

## 6. Ứng dụng thực tiễn & Liên hệ môn học
* **Giải đố logic và lập kế hoạch (CSP & Planning):** Tự động tìm lộ trình giải cho các bài toán không có thứ tự sinh cố định (Sudoku, Zebra, NAE-SAT) mà không cần nhãn thứ tự giải khi huấn luyện; cơ chế chọn ô chắc chắn nhất tương đồng trực tiếp với heuristic MRV trong CSP.
* **Hoàn thiện và điền khuyết mã nguồn (Code Infilling):** Hỗ trợ điền chính xác các đoạn code bị khuyết một dòng hoặc nhiều dòng (như benchmark HumanEval-Infill) bằng cách chốt các token có độ tự tin cao trước.
* **Suy luận toán học (Mathematical Reasoning):** Giải quyết các bài toán tự luận/suy luận đa bước (như benchmark Math trên mô hình LLaDA 8B) có trình tự phân tích phụ thuộc vào từng đề bài cụ thể.
* **Hoàn thiện mạch truyện (Text Infilling):** Điền khuyết văn bản tự nhiên (như tập ROCStories) giúp giảm mạnh Generative Perplexity (tăng độ mạch lạc) trong khi vẫn duy trì độ đa dạng từ vựng (entropy).

---

## 7. Kế hoạch Pilot Study (2 Tuần) & Phân công nhân sự (5 Thành viên)

| Thành viên / Giai đoạn | Phụ trách chính | Mục tiêu đầu ra (Deliverables) |
| :--- | :--- | :--- |
| **Thành viên 1 (TV1)** | Dữ liệu & Solver cổ điển | Sinh $10\text{k}$ mẫu Sudoku $4\times4$, $6\times6$; trích xuất $20\text{k}$ mẫu $9\times9$ Radcliffe; cài đặt Backtracking, $\text{MRV}+\text{FC}$, AC-3. |
| **Thành viên 2 (TV2)** | Huấn luyện MDM & ARM | Xây dựng kiến trúc Tiny Transformer ($\sim 6\text{M}$ tham số), code vòng lặp dynamic masking cho MDM và causal training cho ARM trên Colab. |
| **Thành viên 3 (TV3)** | Module Suy luận thích ứng | Cài đặt $3$ hàm chọn vị trí mở khóa (Random, Top Probability, Top Probability Margin + Gumbel noise). |
| **Thành viên 4 (TV4)** | Đo lường & Phân tích | Viết hàm chấm điểm toàn bảng, từng ô, đếm lỗi ràng buộc, thống kê độ chính xác theo margin cao/thấp. |
| **Thành viên 5 (TV5)** | Tích hợp & Xây dựng Demo | Quản lý repo/notebook chung, làm giao diện hiển thị trực quan thứ tự mở ô theo thời gian (IPython / Gradio). |
| **Tuần 1 (Ngày $1-7$)** | Kiểm tra quy trình & Huấn luyện thử | Đọc hiểu paper + repo mẫu; hoàn tất pipeline dữ liệu; train thành công checkpoint MDM đầu tiên trên Colab Free. |
| **Tuần 2 (Ngày $8-14$)** | Đánh giá chênh lệch & Demo Margin | Chạy so sánh $3$ quy tắc trên cùng checkpoint MDM để xác nhận Margin vượt trội Random; hoàn thiện Demo MDM dùng Margin. |

---

## 8. Dự kiến sản phẩm bàn giao cuối kỳ (Final Deliverables)
* **Mã nguồn & README:** Source code chạy lại được (reproducible), phân tách rõ phần kế thừa từ repo gốc và phần nhóm tự phát triển.
* **Chương trình Demo:** Hiển thị trực quan quá trình giải Sudoku và thứ tự các ô `[MASK]` được mở khóa qua từng bước thời gian.
* **Báo cáo khoa học:** Báo cáo chi tiết kèm bảng số liệu trung bình + độ lệch chuẩn qua các lần chạy và biểu đồ phân tích hiệu năng.