# 📢 THÔNG BÁO CHỐT KIỂM ĐỊNH 14H00 (GIAI ĐOẠN 4 - NGÀY 3)
## KẾT QUẢ ĐÁNH GIÁ THỰC NGHIỆM TV1 & TV3 – YÊU CẦU ĐIỀU CHỈNH GẤP

* **Người lập báo cáo:** Thành viên 4 (TV4 – Phụ trách Đo lường & Metrics)
* **Người nhận:** Toàn bộ thành viên Nhóm 5 (Đặc biệt **TV1**, **TV3**, **TV2**)
* **Thời điểm chốt kiểm định:** 14h00 – Ngày 3 (Giai đoạn 4: Đánh giá chênh lệch)
* **Trạng thái:** ⚠️ **WARNING – ĐỘ CHÊNH LỆCH QUÁ NHỎ (CẦN ĐIỀU CHỈNH GẤP)**

---

## 1. TỔNG QUAN KẾT QUẢ KIỂM ĐỊNH TẠI MỐC 14H00

Theo đúng phân công tại **Task 16**, TV4 đã tiến hành chấm điểm và đối sánh thực nghiệm trên tập Test chuẩn **200 bảng Sudoku $6\times6$** qua **3 random seeds** ($[42, 123, 999]$) giữa kết quả của **TV1** (Task 14 – CSP) và **TV3** (Task 15 – MDM Adaptive Inference).

### Bảng Kết Quả Đối Sánh Toàn Diện

| Nhóm Phương Pháp | Chiến Lược / Thuật Toán | Board Acc (%) | Cell Acc (%) | Lỗi Vi Phạm TB / Bảng | Thời Gian Giải (s/bảng) |
|:---|:---|:---:|:---:|:---:|:---:|
| **CSP Baseline (TV1)** | **Backtracking** (Cổ điển) | **$61.50\%$** | **$93.75\%$** | **$0.00$** | $\approx 0.0003\text{s}$ |
| **CSP Baseline (TV1)** | **MRV + Forward Checking** | **$62.00\%$** | **$93.89\%$** | **$0.00$** | $\approx 0.0003\text{s}$ |
| **CSP Baseline (TV1)** | **AC-3 Propagation** | **$62.00\%$** | **$93.89\%$** | **$0.00$** | $\approx 0.0005\text{s}$ |
| **Masked Diffusion (TV3)** | **MDM – Random Unmasking** | $0.00 \pm 0.00\%$ | $58.24 \pm 0.08\%$ | $29.02 \pm 0.13$ | $\approx 0.0012\text{s}$ |
| **Masked Diffusion (TV3)** | **MDM – Top Probability** | $0.00 \pm 0.00\%$ | $58.05 \pm 0.26\%$ | $29.25 \pm 0.19$ | $\approx 0.0014\text{s}$ |
| **Masked Diffusion (TV3)** | **MDM – Top Prob Margin** | $0.00 \pm 0.00\%$ | $58.28 \pm 0.31\%$ | $29.09 \pm 0.32$ | $\approx 0.0015\text{s}$ |

### Đánh Giá Độ Chênh Lệch Margin vs Random

* **$\Delta\text{Cell Accuracy (Margin - Random)}$:** $\mathbf{+0.04\%}$ (Ngưỡng yêu cầu tối thiểu: $\ge \mathbf{1.0\%}$).
* **$\Delta\text{Board Accuracy (Margin - Random)}$:** $\mathbf{0.00\%}$ (Cả 2 chiến lược chưa giải đúng trọn vẹn bảng nào).
* **Số lỗi vi phạm ràng buộc:** Không giảm đáng kể ($29.09$ lỗi so với $29.02$ lỗi/bảng).
* **High Margin ($\ge 0.5$) vs Low Margin ($< 0.2$):** Độ chính xác chỉ chênh lệch $+0.50\%$ ($17.66\%$ vs $17.16\%$), cho thấy mức độ tự tin (margin confidence) chưa thực sự phân hóa được ô dễ và ô khó.

> [!WARNING]
> **KẾT LUẬN CỦA TV4:** Độ chênh lệch giữa Margin và Random **quá nhỏ** ($\Delta = +0.04\%$). Nếu giữ nguyên cấu hình này để làm báo cáo cuối kỳ, nhóm sẽ **không thể chứng minh được luận điểm cốt lõi của paper ICML 2025** (*Top Probability Margin vượt trội Random*).
> 
> Báo ngay lúc 14h00 để **TV1** và **TV3** (kết hợp **TV2**) thực hiện ngay các điều chỉnh bên dưới!

---

## 2. PHÂN TÍCH NGUYÊN NHÂN CỐT LÕI (ROOT CAUSE ANALYSIS)

1. **Về phía Dữ liệu & Đề Test (TV1):**
   * Đề test $6\times6$ hiện tại đang đục lỗ ngẫu nhiên từ $14-22$ ô trống trên tổng số $36$ ô (tức là còn tới $14-22$ ô gợi ý cho sẵn).
   * Khi số ô gợi ý còn nhiều, việc điền ngẫu nhiên (Random unmasking) vẫn có xác suất ăn may trúng tới $\sim 58\%$ Cell Accuracy, làm cho chiến lược Margin không bộc lộ được ưu thế vượt trội.
2. **Về phía Siêu tham số Suy luận (TV3):**
   * Hệ số nhiễu Gumbel đang đặt cố định là `gumbel_scale = 0.5`. Khi mô hình Pilot có thang logit chưa lớn, nhiễu Gumbel biên độ $0.5$ đã lấn át độ chênh lệch Top-1 trừ Top-2, khiến cơ chế chọn ô của Margin bị ngẫu nhiên hóa gần giống Random.
   * Số bước `steps = 50` cần được tinh chỉnh phù hợp hơn với số ô bị mask thực tế.
3. **Về phía Mô hình & Trọng số (TV2):**
   * Hiện tại chưa có file checkpoint đã hội tụ `checkpoints/mdm_pilot_best.pt` trong thư mục dự án (pipeline đang phải fallback dùng DummyMDM / trọng số khởi tạo ngẫu nhiên). Cần checkpoint thực sự đã học ràng buộc Sudoku để Margin phát huy tác dụng.

---

## 3. CHECKLIST YÊU CẦU ĐIỀU CHỈNH CHI TIẾT

### 🎯 3.1. DÀNH CHO THÀNH VIÊN 1 (TV1 – Dữ liệu & CSP)
* [ ] **Tăng độ khó đề test $6\times6$:**
  * Tăng số lượng ô bị che `[MASK]` (giá trị `0`) từ $14-22$ lên **$22-28$ ô** (chỉ để lại $8-14$ ô cho trước ban đầu). Khi đó Random unmasking sẽ tụt độ chính xác rõ rệt do không thể điền bừa trúng.
  * Cập nhật logic trong file `notebooks/tv1_data_test.ipynb` (hoặc hàm sinh data trong `src/data_csp.py`):
    ```python
    # Gợi ý code điều chỉnh trong tv1_data_test.ipynb:
    num_missing = np.random.randint(22, 29)  # Tăng số ô che để nâng độ khó
    mask_indices = np.random.choice(seq_len, size=num_missing, replace=False)
    quiz_1d[mask_indices] = 0
    ```
* [ ] **Tạo thêm tập test con Sudoku $9\times9$ khó:**
  * Trích xuất các bảng $9\times9$ có từ $20-28$ givens từ tập Radcliffe (những bảng khó bắt buộc phải suy luận logic nhiều bước).
* [ ] **Cập nhật lại file `data/sudoku_pilot.npz`** và chạy lại Cell 4 của `notebooks/tv1_data_test.ipynb` để xuất lại `results/baseline_csp_results.npz`.

---

### 🎯 3.2. DÀNH CHO THÀNH VIÊN 3 (TV3 – Adaptive Inference)
* [ ] **Tinh chỉnh siêu tham số `gumbel_scale` trong `src/inference.py`:**
  * Giảm `gumbel_scale` từ $0.5$ xuống **$0.1$ hoặc $0.2$** (hoặc đặt mặc định $0.1$ cho bảng nhỏ $6\times6$).
  * Việc giảm nhiễu giúp cơ chế Margin tập trung tuyệt đối vào các ô có độ tự tin cao nhất ($\|p_1 - p_2\|$ lớn nhất), loại bỏ tính ngẫu nhiên không mong muốn.
    ```python
    # Gợi ý code điều chỉnh trong src/inference.py (adaptive_decode & run_task15_experiment):
    def adaptive_decode(
        model,
        quiz_1d,
        strategy="margin",
        steps=50,
        gumbel_scale=0.1,  # <--- Giảm từ 0.5 xuống 0.1-0.2
        device="cpu",
    ):
    ```
* [ ] **Tăng số bước suy luận `steps`:**
  * Tăng `steps` từ $50$ lên **$70 - 80$ bước** để mô hình giải mã chậm và chắc, chỉ mở $1-2$ ô có độ tin cậy cao nhất ở mỗi bước thời gian.
* [ ] **Hỗ trợ Grid-search nhanh:** Cho phép truyền danh sách `gumbel_scales = [0.0, 0.1, 0.2, 0.5]` trong `run_task15_experiment` để tìm cấu hình tối ưu nhất.

---

### 🎯 3.3. DÀNH CHO THÀNH VIÊN 2 (TV2 – Huấn luyện MDM)
* [ ] **Đẩy nhanh tiến độ train trên Google Colab:**
  * Chạy huấn luyện mô hình `TinyMDM` trên Colab với dữ liệu $6\times6$ ít nhất $25-40$ epochs.
  * Theo dõi `compute_mdm_loss`: Đảm bảo Loss hội tụ xuống dưới $0.5$ (tối ưu $\le 0.3$).
* [ ] **Lưu và chia sẻ Checkpoint:**
  * Lưu file trọng số tốt nhất vào đường dẫn: `checkpoints/mdm_pilot_best.pt`.
  * Upload lên thư mục Google Drive chung của nhóm hoặc commit nếu dung lượng $< 50\text{MB}$ để TV3 và TV4 nạp vào đánh giá.

---

### 🎯 3.4. DÀNH CHO THÀNH VIÊN 5 (TV5 – Demo & Visualizer)
* [ ] Chuẩn bị sẵn giao diện tại `notebooks/main_demo.ipynb` để nạp file `results/gold_demo_samples.pkl`.
* [ ] Khi TV1 và TV3 cập nhật xong, chạy lại demo để quay clip hoặc hiển thị trực quan animation các ô được mở theo Margin (xanh lá) vs các ô điền bừa (cam).

---

## 4. HƯỚNG DẪN TÁI KIỂM ĐỊNH (RE-AUDIT WORKFLOW)

Sau khi TV1, TV2 và TV3 hoàn thành các điều chỉnh trên, chạy một lệnh duy nhất để TV4 tự động chấm điểm lại toàn bộ và cập nhật chốt kiểm định:

```bash
# Chạy tái kiểm định tự động toàn diện
python src/metrics.py --run_task16
```

### Tiêu Chí Nghiệm Thu Đạt Chuẩn (PASS CRITERIA)
1. **$\Delta\text{Cell Accuracy (Margin - Random)} \ge \mathbf{1.0\%}$** (Kỳ vọng đạt $+5\% \sim +15\%$).
2. **$\Delta\text{Board Accuracy (Margin - Random)} \ge \mathbf{0.5\%}$** (Kỳ vọng Margin giải đúng được ít nhất $10-30\%$ số bảng, trong khi Random $\approx 0\%$).
3. **Lỗi vi phạm ràng buộc:** Chiến lược Margin phải có số lỗi TB thấp hơn rõ rệt so với Random.
4. **Phân hóa Margin:** Nhóm High Margin ($\ge 0.5$) phải có độ chính xác cao hơn nhóm Low Margin ($< 0.2$) ít nhất $10\%$.

---
*Thông báo được gửi chính thức bởi TV4 lúc 14h00. Đề nghị các thành viên xác nhận và tiến hành cập nhật trước 17h00 cùng ngày!*
