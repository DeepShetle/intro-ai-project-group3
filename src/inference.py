"""
src/inference.py – TV3: Adaptive Decoding Strategies for Masked Diffusion Models
================================================================================
Dựa trên bài báo ICML 2025:
"Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions"

File chịu trách nhiệm:
  1. sample_gumbel: Sinh nhiễu Gumbel chuẩn để tạo tính ngẫu nhiên khám phá.
  2. compute_k_unmask: Lịch trình tính số ô K cần mở khóa mỗi bước (Phụ lục D.1.2).
  3. adaptive_decode: Vòng lặp giải mã thích ứng ngược (reverse sampling) với 3 chiến lược:
     - "random": Mở ngẫu nhiên (Vanilla MDM baseline).
     - "top_prob": Mở theo xác suất lớn nhất (max prob).
     - "margin": Mở theo hiệu số Top-1 trừ Top-2 (Top prob margin + Gumbel noise).
  4. run_inference_on_dataset: Chạy suy luận hàng loạt trên tập test để khớp với TV4.
"""

from __future__ import annotations

import sys
try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional


def sample_gumbel(
    shape: torch.Size | tuple | list,
    device: torch.device | str = "cpu",
    eps: float = 1e-10,
) -> torch.Tensor:
    """Sinh nhiễu phân phối Gumbel chuẩn: -log(-log(U + eps) + eps).

    Parameters
    ----------
    shape : torch.Size | tuple | list
        Kích thước tensor nhiễu cần sinh.
    device : torch.device | str
        Thiết bị ('cpu' hoặc 'cuda').
    eps : float
        Hằng số nhỏ chống tràn số/log(0).

    Returns
    -------
    gumbel_noise : torch.Tensor
        Tensor nhiễu Gumbel cùng shape và device.
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def compute_k_unmask(
    num_masked: int,
    step_idx: int,
    actual_steps: int,
    schedule: str = "linear",
) -> int:
    """Tính số lượng token K cần mở khóa tại bước hiện tại theo Phụ lục D.1.2.

    Với lịch trình linear:
      remaining_steps = actual_steps - step_idx
      K = max(1, round(num_masked / remaining_steps))
    Đảm bảo mở hết toàn bộ ô còn lại ở bước cuối cùng và không bao giờ vượt quá num_masked.

    Parameters
    ----------
    num_masked : int
        Số ô còn đang bị che [MASK] (giá trị 0) hiện tại.
    step_idx : int
        Chỉ số bước lặp hiện tại (0-indexed: 0, 1, ..., actual_steps - 1).
    actual_steps : int
        Tổng số bước suy luận thực tế.
    schedule : str
        Lịch trình mở ô ('linear').

    Returns
    -------
    k : int
        Số lượng ô cần unmask tại bước này.
    """
    if num_masked <= 0:
        return 0

    remaining_steps = actual_steps - step_idx
    # Bước cuối cùng hoặc chỉ còn 1 bước -> mở hết tất cả các ô còn lại
    if remaining_steps <= 1:
        return int(num_masked)

    if schedule == "linear":
        k = max(1, int(round(num_masked / remaining_steps)))
    else:
        k = max(1, int(round(num_masked / remaining_steps)))

    return int(min(k, num_masked))


@torch.no_grad()
def adaptive_decode(
    model: nn.Module,
    quiz_1d: np.ndarray,
    strategy: str = "margin",
    steps: int = 50,
    gumbel_scale: float = 0.5,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[Dict]]:
    """Giải mã thích ứng bảng Sudoku bằng Masked Diffusion Model (MDM).

    Quy trình:
      1. Đặt model ở chế độ eval() và no_grad().
      2. Xác định các ô bị che (0 = [MASK]). Tính số bước thực tế actual_steps.
      3. Lặp qua từng bước:
         - Đưa x_t qua model lấy logits, gán logits[:, 0] = -1e9 để triệt tiêu token [MASK].
         - Tính xác suất softmax và lấy Top-2 xác suất cao nhất.
         - Tính điểm chọn vị trí (scores) CHỈ trên các ô đang bị che theo strategy:
             + 'random'   : rand()
             + 'top_prob' : top1_prob + gumbel_noise
             + 'margin'   : (top1_prob - top2_prob) + gumbel_noise
         - Chọn K ô có điểm cao nhất, cập nhật giá trị dự đoán vào x_t.
         - Ghi nhận vào step_history với đúng 5 khóa chuẩn của Data Contract.

    Parameters
    ----------
    model : nn.Module
        Mô hình MDM đã huấn luyện (hoặc DummyMDM để test).
    quiz_1d : np.ndarray, shape (L,), dtype int64
        Bảng đề 1D (0 = [MASK], 1..N = giá trị đã cho).
    strategy : str
        Chiến lược chọn thứ tự mở ô: 'random', 'top_prob', 'margin'.
    steps : int
        Số bước giải mã tối đa (mặc định 50).
    gumbel_scale : float
        Hệ số nhiễu Gumbel (mặc định 0.5 theo Phụ lục D.2 của paper).
    device : str
        Thiết bị thực thi ('cpu' hoặc 'cuda').

    Returns
    -------
    pred_board_1d : np.ndarray, shape (L,), dtype int64
        Bảng kết quả sau khi giải xong (không còn ô 0).
    step_history : List[Dict]
        Lịch sử từng bước giải. Mỗi phần tử có đúng 5 khóa:
          - 'step'            : int
          - 'unmasked_indices': List[int]
          - 'values'          : List[int]
          - 'margins'         : List[float] (raw margins trước khi cộng Gumbel)
          - 'board_state'     : np.ndarray shape (L,)
    """
    model.eval()
    model = model.to(device)

    # Khởi tạo tensor trạng thái x_t từ quiz_1d
    x_t = torch.tensor(quiz_1d, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, L)
    initial_masks = int((x_t[0] == 0).sum().item())

    step_history: List[Dict] = []

    # Nếu bảng không có ô nào bị che, trả về ngay
    if initial_masks == 0:
        return quiz_1d.copy().astype(np.int64), step_history

    # Số bước thực tế: tối đa là số ô bị che
    actual_steps = min(steps, initial_masks) if initial_masks > 0 else 1

    for step_idx in range(actual_steps):
        # 1. Tìm vị trí các ô đang bị che [MASK] (giá trị 0)
        mask_positions = torch.where(x_t[0] == 0)[0]
        num_masked = len(mask_positions)
        if num_masked == 0:
            break

        # 2. Forward pass lấy logits
        logits = model(x_t)  # shape (1, L, vocab_size)
        logits_1d = logits[0].clone()  # shape (L, vocab_size)

        # Ép logits của token [MASK] (id = 0) về cực âm lớn để không bao giờ dự đoán lại [MASK]
        logits_1d[:, 0] = -1e9

        # Tính xác suất phân phối qua softmax
        probs = torch.softmax(logits_1d, dim=-1)  # shape (L, vocab_size)

        # 3. Lấy Top-2 xác suất cao nhất tại mỗi vị trí
        top2_vals, top2_idx = torch.topk(probs, k=2, dim=-1)  # shapes (L, 2)
        top1_prob = top2_vals[:, 0]        # shape (L,)
        top2_prob = top2_vals[:, 1]        # shape (L,)
        pred_tokens = top2_idx[:, 0]       # shape (L,)
        raw_margins = top1_prob - top2_prob  # shape (L,)

        # 4. Tính điểm chọn vị trí (scores) CHỈ trên các ô đang bị che
        if strategy == "random":
            scores = torch.rand(num_masked, device=device)
        elif strategy == "top_prob":
            scores = top1_prob[mask_positions]
            if gumbel_scale > 0:
                scores = scores + gumbel_scale * sample_gumbel(scores.shape, device=device, eps=1e-10)
        elif strategy == "margin":
            scores = raw_margins[mask_positions]
            if gumbel_scale > 0:
                scores = scores + gumbel_scale * sample_gumbel(scores.shape, device=device, eps=1e-10)
        else:
            raise ValueError(
                f"Chiến lược không hợp lệ: '{strategy}'. "
                f"Các chiến lược được hỗ trợ: 'random', 'top_prob', 'margin'."
            )

        # 5. Xác định số lượng ô K cần mở khóa ở bước này
        k = compute_k_unmask(
            num_masked=num_masked,
            step_idx=step_idx,
            actual_steps=actual_steps,
            schedule="linear",
        )
        k = min(k, num_masked)

        # 6. Chọn K vị trí có score cao nhất trong mask_positions
        topk_rel_indices = torch.topk(scores, k=k).indices
        chosen_indices = mask_positions[topk_rel_indices]
        chosen_values = pred_tokens[chosen_indices]
        # Chú ý: lưu raw_margins (trước khi cộng nhiễu Gumbel) theo Hợp đồng Dữ liệu
        chosen_margins = raw_margins[chosen_indices]

        # 7. Cập nhật giá trị dự đoán vào x_t
        x_t[0, chosen_indices] = chosen_values

        # 8. Ghi lại lịch sử bước giải (đúng 5 khóa chuẩn, step bắt đầu từ 1)
        step_history.append({
            "step": int(len(step_history) + 1),
            "unmasked_indices": chosen_indices.cpu().tolist(),
            "values": chosen_values.cpu().tolist(),
            "margins": [round(float(m), 4) for m in chosen_margins.cpu().tolist()],
            "board_state": x_t[0].cpu().numpy().copy().astype(np.int64),
        })

    pred_board_1d = x_t[0].cpu().numpy().astype(np.int64)
    return pred_board_1d, step_history


def run_inference_on_dataset(
    model: nn.Module,
    quizzes_2d: np.ndarray,
    strategy: str = "margin",
    steps: int = 50,
    gumbel_scale: float = 0.5,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[List[Dict]]]:
    """Chạy suy luận adaptive decoding trên một tập đề bài Sudoku (Batch/Dataset).

    Hàm phục vụ Ngày 2: Đầu ra khớp trực tiếp với evaluate_predictions(...) của TV4.

    Parameters
    ----------
    model : nn.Module
        Mô hình MDM (hoặc DummyMDM).
    quizzes_2d : np.ndarray, shape (Num_samples, L), dtype int64
        Mảng 2D chứa các bảng đề Sudoku.
    strategy : str
        Chiến lược giải: 'random', 'top_prob', 'margin'.
    steps : int
        Số bước giải mã tối đa.
    gumbel_scale : float
        Hệ số nhiễu Gumbel.
    device : str
        Thiết bị chạy model ('cpu' hoặc 'cuda').

    Returns
    -------
    preds_2d : np.ndarray, shape (Num_samples, L), dtype int64
        Mảng 2D chứa các bảng lời giải dự đoán.
    all_histories : List[List[Dict]]
        Danh sách các step_history tương ứng từng bảng.
    """
    num_samples = len(quizzes_2d)
    preds_list: List[np.ndarray] = []
    all_histories: List[List[Dict]] = []

    for i in range(num_samples):
        quiz_1d = quizzes_2d[i]
        pred_1d, history = adaptive_decode(
            model=model,
            quiz_1d=quiz_1d,
            strategy=strategy,
            steps=steps,
            gumbel_scale=gumbel_scale,
            device=device,
        )
        preds_list.append(pred_1d)
        all_histories.append(history)

    preds_2d = np.array(preds_list, dtype=np.int64)
    return preds_2d, all_histories


# ---------------------------------------------------------------------------
# Unit Test tự động với DummyMDM và Mock Data (Kiểm thử Ngày 1 & Ngày 2)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from src.interfaces import (
        DummyMDM,
        generate_mock_data,
        validate_step_history,
    )

    print("=" * 70)
    print("KIỂM THỬ MODULE INFERENCE (TV3) - ADAPTIVE DECODING STRATEGIES")
    print("=" * 70)

    # 1. Sinh dữ liệu kiểm thử
    print("\n[BƯỚC 1] Khởi tạo DummyMDM và Mock Data (9x9, L=81)")
    vocab_size = 10  # 0..9 (0 là MASK, 1..9 là giá trị)
    model = DummyMDM(vocab_size=vocab_size)
    quizzes, solutions = generate_mock_data(num_samples=5, n_size=9)
    sample_quiz = quizzes[0]
    num_masks = int(np.sum(sample_quiz == 0))
    print(f"  - Quiz shape: {sample_quiz.shape}, Số ô MASK ban đầu: {num_masks} / 81")

    # 2. Kiểm thử 3 chiến lược: 'random', 'top_prob', 'margin'
    strategies = ["random", "top_prob", "margin"]
    test_results = {}

    print("\n[BƯỚC 2] Chạy kiểm thử adaptive_decode cho cả 3 chiến lược...")
    for strat in strategies:
        print(f"\n--- Kiểm thử chiến lược: '{strat}' ---")
        pred_board, history = adaptive_decode(
            model=model,
            quiz_1d=sample_quiz,
            strategy=strat,
            steps=50,
            gumbel_scale=0.5,
            device="cpu",
        )

        # Kiểm tra 1: Bảng đầu ra có đúng shape (81,) và không còn ô nào bằng 0
        assert pred_board.shape == (81,), f"Shape sai: {pred_board.shape}"
        assert not np.any(pred_board == 0), f"Chiến lược '{strat}': Vẫn còn ô MASK (0) trong bảng kết quả!"
        print(f"  [OK] Bảng dự đoán hợp lệ (shape: {pred_board.shape}, không còn ô 0).")

        # Kiểm tra 2: step_history có đầy đủ 5 keys chuẩn và thỏa Data Contract
        is_valid = validate_step_history(history, seq_len=81)
        assert is_valid, f"Chiến lược '{strat}': step_history không thỏa mãn Data Contract!"
        print(f"  [OK] step_history hợp lệ với {len(history)} bước giải.")

        # Kiểm tra số lượng unmasked ô khớp tổng số MASK
        total_unmasked = sum(len(step["unmasked_indices"]) for step in history)
        assert total_unmasked == num_masks, (
            f"Tổng số ô unmasked ({total_unmasked}) != số MASK ban đầu ({num_masks})"
        )
        print(f"  [OK] Tổng số ô đã mở: {total_unmasked} khớp chính xác với {num_masks} ô MASK.")

        test_results[strat] = history

    # 3. In ra 2 bước đầu tiên của step_history ở chiến lược 'margin'
    print("\n" + "=" * 70)
    print("[BƯỚC 3] 2 BƯỚC ĐẦU TIÊN CỦA STEP_HISTORY (Chiến lược 'margin'):")
    print("=" * 70)
    margin_history = test_results["margin"]
    for step_info in margin_history[:2]:
        print(f"Bước {step_info['step']}:")
        print(f"  - unmasked_indices : {step_info['unmasked_indices']}")
        print(f"  - values           : {step_info['values']}")
        print(f"  - margins (raw)    : {step_info['margins']}")
        print(f"  - board_state shape: {step_info['board_state'].shape}")
        print(f"  - board_state (sample 18 ô đầu): {step_info['board_state'][:18]}")
        print("-" * 50)

    # 4. Kiểm thử hàm run_inference_on_dataset (Batch test)
    print("\n[BƯỚC 4] Kiểm thử run_inference_on_dataset trên tập batch 5 mẫu...")
    batch_preds, batch_histories = run_inference_on_dataset(
        model=model,
        quizzes_2d=quizzes,
        strategy="margin",
        steps=50,
        gumbel_scale=0.5,
        device="cpu",
    )
    assert batch_preds.shape == (5, 81), f"batch_preds shape sai: {batch_preds.shape}"
    assert len(batch_histories) == 5, f"Số lượng histories sai: {len(batch_histories)}"
    assert not np.any(batch_preds == 0), "Vẫn còn ô 0 trong batch_preds!"
    print(f"  [OK] run_inference_on_dataset thành công: shape={batch_preds.shape}, {len(batch_histories)} histories.")

    # 5. Kiểm thử trường hợp Sudoku 4x4 (L=16)
    print("\n[BƯỚC 5] Kiểm thử bảng Sudoku 4x4 (L=16, vocab_size=5)...")
    model_4x4 = DummyMDM(vocab_size=5)
    q4, _ = generate_mock_data(num_samples=2, n_size=4)
    pred_4x4, hist_4x4 = adaptive_decode(
        model=model_4x4,
        quiz_1d=q4[0],
        strategy="margin",
        steps=20,
        gumbel_scale=0.5,
    )
    assert pred_4x4.shape == (16,), f"Shape 4x4 sai: {pred_4x4.shape}"
    assert not np.any(pred_4x4 == 0), "4x4 vẫn còn ô 0!"
    assert validate_step_history(hist_4x4, seq_len=16), "step_history 4x4 không hợp lệ!"
    print(f"  [OK] Sudoku 4x4 giải mã thành công trong {len(hist_4x4)} bước.")

    print("\n" + "=" * 70)
    print("TẤT CẢ UNIT TESTS CỦA TV3 ĐÃ PASS 100%! SẴN SÀNG NGHIỆM THU.")
    print("=" * 70)
