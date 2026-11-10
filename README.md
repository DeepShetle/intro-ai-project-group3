# 🧩 Nghiên cứu Chiến lược Suy luận Thích ứng (Adaptive Inference) trong Masked Diffusion Models cho Sudoku

> **Đề tài Pilot Study (Nhóm 5 - Intro to AI)**  
> Dựa trên bài báo ICML 2025: *"Train for the Worst, Plan for the Best: Adaptive Inference in Masked Diffusion Models"*.

---

## 📌 1. Giới thiệu đề tài
Dự án nghiên cứu và hiện thực hóa các chiến lược giải mã thích ứng (Adaptive Decoding Strategies) cho mô hình Masked Diffusion Models (MDM) trong bài toán giải đố Sudoku logic ($4 \times 4, 6 \times 6, 9 \times 9$). Trọng tâm là so sánh các cơ chế:
1. **Random unmasking**: Mở ô ngẫu nhiên không trọng số.
2. **Top Probability**: Mở ô có xác suất dự đoán cao nhất.
3. **Top Probability Margin + Gumbel Noise**: Mở ô dựa trên khoảng cách giữa 2 xác suất cao nhất (Margin confidence) kèm nhiễu Gumbel kiểm soát entropy.
4. **Baseline CSP cổ điển**: Backtracking, MRV + Forward Checking, AC-3.

## ⚖️ Bảng Phân tách Phần Kế thừa vs Phần Nhóm Tự Phát triển

| Hạng mục / Module | Kế thừa từ Repo gốc (Ye et al., ICML 2025) | Nhóm tự nghiên cứu & Hiện thực mới 100% |
|:---|:---|:---|
| **Lý thuyết & Thuật toán** | Ý tưởng thứ tự giải mã thích ứng, công thức Margin Confidence, hệ số nhiễu Gumbel ($0.5$). | Chuyển đổi toàn bộ từ miền xử lý ngôn ngữ tự nhiên (NLP) sang cấu trúc ma trận giải đố logic Sudoku ($4\times4, 6\times6, 9\times9$). |
| **Kiến trúc Mạng (TV2)** | Khái niệm Bidirectional Masked Transformer. | Tự thiết kế `TinyMDM` nhỏ gọn (~270k - 6M params), cơ chế nhúng 1D Positional Embedding cho Sudoku tối ưu cho Colab Free. |
| **Baseline Đối sánh (TV1)**| Không có (Paper gốc chỉ so sánh MDM với ARM). | Xây dựng bộ ba thuật toán tìm kiếm thỏa mãn ràng buộc (CSP): Backtracking, MRV + Forward Checking, AC-3. |
| **Giao thức Dữ liệu (TV5)**| Không có. | Chuẩn hóa Data Contract (`step_history` 5 khóa), xây dựng cổng kiểm định tự động `validate_step_history`. |
| **Giao diện Demo (TV5)**   | Không có. | Toàn bộ giao diện tương tác HTML/CSS, thanh trượt thời gian, hiệu ứng animation từng bước trên Colab/Jupyter. |
| **Đo lường & Phân tích (TV4)**| Thước đo độ chính xác chuỗi chung. | Hàm đếm vi phạm ràng buộc Sudoku `count_sudoku_violations`, phân tích tương quan Margin vs Cell Accuracy qua nhiều random seeds. |

---

## 📂 2. Cấu trúc thư mục dự án

```text
intro-ai-project-group3/
├── .gitignore
├── requirements.txt
├── README.md
├── notebooks/
│   ├── main_demo.ipynb          # TV5: Giao diện Demo tương tác trực quan hóa mở ô
│   ├── tv1_data_test.ipynb      # TV1: Notebook kiểm thử dữ liệu & CSP
│   └── tv2_train_colab.ipynb    # TV2: Huấn luyện TinyMDM trên Colab
└── src/
    ├── __init__.py
    ├── interfaces.py            # TV5: Khóa cứng Data Contract, DummyMDM & Mock Data
    ├── visualizer.py            # TV5: Xuất HTML lưới Sudoku & phát lại step_history
    ├── data_csp.py              # TV1: Bộ nạp dữ liệu Sudoku & bộ giải CSP
    ├── models.py                # TV2: Kiến trúc mạng TinyMDM
    ├── inference.py             # TV3: Các thuật toán giải mã thích ứng (Adaptive Decode)
    └── metrics.py               # TV4: Hàm đo lường độ chính xác và vi phạm ràng buộc
```

---

## 👥 3. Phân công vai trò thành viên

| Thành viên | Trách nhiệm chính | File phụ trách |
|---|---|---|
| **TV1** | Dữ liệu & CSP Baseline Solver | `src/data_csp.py`, `notebooks/tv1_data_test.ipynb` |
| **TV2** | Kiến trúc mô hình & Huấn luyện MDM | `src/models.py`, `notebooks/tv2_train_colab.ipynb` |
| **TV3** | Giải mã thích ứng (Adaptive Inference) | `src/inference.py` |
| **TV4** | Đánh giá & Phân tích số liệu (Metrics) | `src/metrics.py` |
| **TV5** | Quản lý Repo, Tích hợp, Demo Trực quan hóa | `src/interfaces.py`, `src/visualizer.py`, `notebooks/main_demo.ipynb` |

---

## 📜 4. Hợp đồng Dữ liệu (Data Contract Specification)

Tất cả các thành viên tuân thủ nghiêm ngặt chuẩn giao tiếp dữ liệu sau:

### 4.1. Vocabulary & Shape
- `0`: Ký hiệu ô trống (`[MASK]`).
- `1..N`: Các chữ số hợp lệ trên bảng Sudoku ($N \in \{4, 6, 9\}$).
- Mọi bảng Sudoku truyền qua hàm đều ở định dạng **mảng 1D `np.ndarray`** kiểu `int64` có chiều dài $L = N^2$ ($16, 36, 81$), trải phẳng từ trái sang phải, từ trên xuống dưới.

### 4.2. Định dạng `step_history`
Mọi hàm giải đố (`solve_csp` của TV1 và `adaptive_decode` của TV3) đều trả về tuple:
```python
(pred_board_1d, step_history)
```
Trong đó, `step_history` là `List[Dict]`. Mỗi phần tử đại diện cho một bước mở ô và bắt buộc có đúng **5 khóa**:
```python
{
    "step": int,                 # Số thứ tự bước (1, 2, 3...)
    "unmasked_indices": list[int],# Vị trí các ô được gỡ mask tại bước này
    "values": list[int],          # Giá trị gán cho các ô tương ứng
    "margins": list[float],       # Độ tin cậy (Margin confidence)
    "board_state": np.ndarray     # Trạng thái toàn bộ bảng sau bước này (shape (L,))
}
```

---

## 🎨 5. Quy tắc Trực quan hóa (Visualizer Color Legend)

Trên giao diện web/notebook tại `notebooks/main_demo.ipynb` và `src/visualizer.py`:
- 🟦 **Xanh đậm (`#1a5276`)**: Ô đề bài cho sẵn ban đầu (`quiz_1d != 0`).
- 🟩 **Xanh lá (`#27ae60`)**: Ô vừa được mở ở bước hiện tại với **độ tự tin cao** ($\text{Margin} \ge 0.5$).
- 🟧 **Cam (`#e67e22`)**: Ô vừa được mở ở bước hiện tại với **độ tự tin thấp** ($\text{Margin} < 0.5$).
- ⬜ **Xám nhạt (`#d5dbdb`)**: Ô đã được mở ở các bước trước đó.
- ◻️ **Trắng (`#fdfefe`)**: Ô vẫn còn đang bị che (`[MASK]`).

---

## 🚀 6. Hướng dẫn Cài đặt & Chạy Demo

### 6.1. Môi trường Local
```bash
# Clone repository
git clone https://github.com/<YOUR_ORG>/intro-ai-project-group3.git
cd intro-ai-project-group3

# Tạo và kích hoạt virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt

# Kiểm tra self-test module interfaces & visualizer
python src/interfaces.py
python src/visualizer.py
```

### 6.2. Chạy Interactive Demo trên Google Colab / Jupyter
Mở file `notebooks/main_demo.ipynb`:
1. **Ngày 2**: Chạy mặc định với `USE_REAL_MODEL = False` để thử nghiệm mượt mà giao diện Widget/Thanh trượt cùng `DummyMDM` và dữ liệu mock.
2. **Ngày 3+**: Chuyển cờ `USE_REAL_MODEL = True` để nạp trọng số `tiny_mdm.pt` từ Google Drive và chạy giải đố bằng thuật toán suy luận thích ứng thật (`adaptive_decode`).
