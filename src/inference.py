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
  5. load_pretrained_mdm: Nạp checkpoint mô hình TinyMDM (hoặc gọi từ src.models).
  6. run_task15_experiment: Thực thi toàn diện Task 15 (xuất kết quả Task 17, Task 20/21).
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

import os
import time
import pickle
import argparse
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
         - Ép logits của các chữ số vượt quá n_size (nếu có) về -1e9.
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
    n_size = int(np.sqrt(x_t.shape[1]))

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
        # Nếu model có vocab_size lớn hơn n_size + 1 (ví dụ model vocab=10 cho Sudoku 6x6),
        # ép logits các chữ số vượt quá n_size về -1e9 để đảm bảo miền giá trị hợp lệ 1..N
        if logits_1d.shape[-1] > n_size + 1:
            logits_1d[:, n_size + 1:] = -1e9

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


def load_pretrained_mdm(
    ckpt_path: str,
    device: str = "cpu",
    vocab_size: Optional[int] = None,
    seq_len: Optional[int] = None,
) -> nn.Module:
    """Nạp checkpoint mô hình TinyMDM từ file .pt.

    Ưu tiên gọi load_pretrained_mdm từ src.models (nếu có).
    Dự phòng an toàn: Khởi tạo TinyMDM từ src.models, tự xử lý cả 2 định dạng
    checkpoint (chứa 'model_state_dict' hoặc lưu trực tiếp state_dict).

    Parameters
    ----------
    ckpt_path : str
        Đường dẫn tới file checkpoint .pt.
    device : str
        Thiết bị tải model ('cpu' hoặc 'cuda').
    vocab_size : Optional[int]
        Kích thước vocab dự kiến.
    seq_len : Optional[int]
        Độ dài chuỗi L dự kiến.

    Returns
    -------
    model : nn.Module
        Mô hình đã load weights và ở chế độ eval().
    """
    try:
        from src.models import load_pretrained_mdm as _models_load_mdm
        return _models_load_mdm(ckpt_path, device=device, vocab_size=vocab_size, seq_len=seq_len)
    except (ImportError, AttributeError):
        pass

    from src.models import TinyMDM

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Không nhận diện được định dạng checkpoint: {type(ckpt)}")

    # Tự động suy luận kích thước từ weights đã lưu
    ckpt_vocab = state_dict.get("token_embedding.weight", None)
    ckpt_pos = state_dict.get("pos_embedding", None)
    actual_vocab = ckpt_vocab.shape[0] if ckpt_vocab is not None else (vocab_size or 10)
    actual_seq = ckpt_pos.shape[1] if ckpt_pos is not None else (seq_len or 81)

    try:
        model = TinyMDM(vocab_size=actual_vocab, seq_len=actual_seq)
        model.load_state_dict(state_dict)
    except Exception:
        model = TinyMDM(vocab_size=vocab_size or 10, seq_len=seq_len or 81)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model


def run_task15_experiment(
    shared_dir: str = ".",
    strategies: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    quick_test: bool = False,
    device: str = "cpu",
) -> Tuple[Dict, List[Dict]]:
    """Chạy thực nghiệm Task 15 toàn diện (xuất dữ liệu chuẩn cho Task 17 & Task 21).

    Tuân thủ nghiêm ngặt Data Contract Ngày 3 & 4:
      1. Đọc data/sudoku_pilot.npz, cắt đúng 200 bảng cuối làm test set:
         test_inputs = quizzes[-200:].astype(np.int64)
         test_targets = solutions[-200:].astype(np.int64)
         (Tuyệt đối không shuffle)
      2. Nạp checkpoint checkpoints/mdm_pilot_best.pt (hoặc DummyMDM nếu chưa có file).
      3. Chạy 3 chiến lược ('random', 'top_prob', 'margin') qua 3 seeds ([42, 123, 999]).
      4. Đóng gói mdm_results có đúng 5 khóa chuẩn và lưu vào results/mdm_inference_results.pkl.
      5. Lọc tối đa 5 gold demo samples (ở seed=42) và lưu vào results/gold_demo_samples.pkl.

    Parameters
    ----------
    shared_dir : str
        Thư mục gốc chứa data/, checkpoints/, results/.
    strategies : Optional[List[str]]
        Danh sách chiến lược cần chạy (mặc định ['random', 'top_prob', 'margin']).
    seeds : Optional[List[int]]
        Danh sách random seeds (mặc định [42, 123, 999]).
    quick_test : bool
        Nếu True, chỉ chạy 1 seed (42) trên 20 mẫu để kiểm tra nhanh pipeline.
    device : str
        Thiết bị chạy inference ('cpu' hoặc 'cuda').

    Returns
    -------
    mdm_results : Dict
        Dictionary kết quả thực nghiệm hoàn chỉnh đúng chuẩn Task 17.
    gold_samples : List[Dict]
        Danh sách 3-5 mẫu Gold Demo phục vụ visualizer của TV5.
    """
    print("=" * 70)
    print("BẮT ĐẦU THỰC NGHIỆM TASK 15: ADAPTIVE INFERENCE ON PILOT DATASET")
    print("=" * 70)

    # 1. Tìm và đọc dữ liệu pilot
    npz_candidates = [
        os.path.join(shared_dir, "data", "sudoku_pilot.npz"),
        os.path.join(shared_dir, "sudoku_pilot.npz"),
        os.path.join(".", "data", "sudoku_pilot.npz"),
        os.path.join(".", "sudoku_pilot.npz"),
    ]
    npz_path = None
    for p in npz_candidates:
        if os.path.isfile(p):
            npz_path = p
            break

    if npz_path is not None:
        print(f"[*] Đang tải dữ liệu từ: {npz_path}")
        try:
            from src.data_csp import load_pilot_data
            quizzes, solutions = load_pilot_data(npz_path)
        except Exception:
            data = np.load(npz_path)
            quizzes = data["quizzes"].astype(np.int64)
            solutions = data["solutions"].astype(np.int64)

        # CẮT ĐÚNG 200 BẢNG CUỐI CÙNG (TUYỆT ĐỐI KHÔNG SHUFFLE)
        test_inputs = quizzes[-200:].astype(np.int64)
        test_targets = solutions[-200:].astype(np.int64)
    else:
        print("[!] CẢNH BÁO: Không tìm thấy file sudoku_pilot.npz! Sử dụng mock data để test luồng.")
        from src.interfaces import generate_mock_data
        mock_q, mock_s = generate_mock_data(num_samples=210, n_size=6)
        test_inputs = mock_q[-200:].astype(np.int64)
        test_targets = mock_s[-200:].astype(np.int64)

    seq_len = int(test_inputs.shape[1])
    n_size = int(np.sqrt(seq_len))
    vocab_size = n_size + 1
    print(f"[*] Tập Test: shape={test_inputs.shape}, seq_len={seq_len}, n_size={n_size}x{n_size}")

    if quick_test:
        print("[*] Chế độ QUICK TEST được kích hoạt: chỉ chạy 20 mẫu đầu của Test set.")
        test_inputs = test_inputs[:20]
        test_targets = test_targets[:20]

    # 2. Tìm và nạp checkpoint mô hình
    ckpt_candidates = [
        os.path.join(shared_dir, "checkpoints", "mdm_pilot_best.pt"),
        os.path.join(shared_dir, "mdm_pilot_best.pt"),
        os.path.join(".", "checkpoints", "mdm_pilot_best.pt"),
        os.path.join(".", "mdm_pilot_best.pt"),
    ]
    ckpt_path = None
    for p in ckpt_candidates:
        if os.path.isfile(p):
            ckpt_path = p
            break

    if ckpt_path is not None:
        print(f"[*] Đang nạp checkpoint từ: {ckpt_path}")
        model = load_pretrained_mdm(
            ckpt_path=ckpt_path,
            device=device,
            vocab_size=vocab_size,
            seq_len=seq_len,
        )
    else:
        print("[!] CẢNH BÁO: Không tìm thấy mdm_pilot_best.pt! Dùng DummyMDM để test định dạng.")
        from src.interfaces import DummyMDM
        model = DummyMDM(vocab_size=vocab_size).to(device)
        model.eval()

    # 3. Thiết lập chiến lược và seeds
    strategies = strategies or ["random", "top_prob", "margin"]
    if seeds is None:
        seeds = [42] if quick_test else [42, 123, 999]

    print(f"[*] Danh sách chiến lược: {strategies}")
    print(f"[*] Danh sách seeds: {seeds}")

    mdm_results: Dict = {
        "inputs": test_inputs,
        "targets": test_targets,
        "random": {},
        "top_prob": {},
        "margin": {},
    }

    validation_verified = False

    for strat in strategies:
        print(f"\n>>> BẮT ĐẦU CHIẾN LƯỢC: [{strat.upper()}] <<<")
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)

            t0 = time.time()
            preds_list = []
            histories_list = []

            for i in range(len(test_inputs)):
                p_1d, h_list = adaptive_decode(
                    model=model,
                    quiz_1d=test_inputs[i],
                    strategy=strat,
                    steps=50,
                    gumbel_scale=0.5,
                    device=device,
                )
                preds_list.append(p_1d)
                histories_list.append(h_list)

                # Kiểm tra Data Contract tại mẫu đầu tiên
                if not validation_verified:
                    try:
                        from src.interfaces import validate_step_history
                        is_valid = validate_step_history(h_list, seq_len=seq_len)
                        assert is_valid, "step_history vi phạm Data Contract!"
                    except Exception as e:
                        # Fallback kiểm tra 5 keys
                        if len(h_list) > 0:
                            req_keys = {"step", "unmasked_indices", "values", "margins", "board_state"}
                            assert set(h_list[0].keys()) == req_keys, "Thiếu khóa trong step_history!"
                    validation_verified = True

            solve_time = time.time() - t0
            preds_arr = np.array(preds_list, dtype=np.int64)

            # Tính các chỉ số nhanh
            board_acc = np.mean([np.array_equal(p, t) for p, t in zip(preds_arr, test_targets)]) * 100.0
            cell_acc = np.mean(preds_arr == test_targets) * 100.0

            mdm_results[strat][seed] = {
                "preds": preds_arr,
                "histories": histories_list,
                "solve_time": float(round(solve_time, 4)),
            }

            print(
                f"  - Seed {seed:3d}: Thời gian={solve_time:6.2f}s "
                f"({solve_time / len(test_inputs):.3f}s/mẫu) | "
                f"Board Acc={board_acc:5.2f}% | Cell Acc={cell_acc:5.2f}%"
            )

    # 4. Xuất file 1: results/mdm_inference_results.pkl
    results_dir = os.path.join(shared_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, "mdm_inference_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(mdm_results, f)
    print(f"\n[OK] Đã lưu File 1: {results_file}")

    # 5. Lọc và xuất file 2: results/gold_demo_samples.pkl
    gold_samples: List[Dict] = []
    ref_seed = 42 if 42 in seeds else seeds[0]

    if "margin" in mdm_results and "random" in mdm_results:
        m_preds = mdm_results["margin"][ref_seed]["preds"]
        m_hists = mdm_results["margin"][ref_seed]["histories"]
        r_preds = mdm_results["random"][ref_seed]["preds"]
        t_preds = mdm_results["top_prob"][ref_seed]["preds"] if "top_prob" in mdm_results else m_preds

        # Điều kiện nghiêm ngặt: Margin đúng 100% toàn bảng, Random giải sai
        strict_gold_indices = [
            idx for idx in range(len(test_inputs))
            if np.array_equal(m_preds[idx], test_targets[idx])
            and not np.array_equal(r_preds[idx], test_targets[idx])
        ]

        if len(strict_gold_indices) > 0:
            chosen_indices = strict_gold_indices[:5]
            print(f"[*] Tìm thấy {len(strict_gold_indices)} mẫu Gold nghiêm ngặt (Margin 100% đúng, Random sai). Lấy {len(chosen_indices)} mẫu.")
        else:
            # Dự phòng khi model pilot chưa hội tụ hoàn toàn (Board Acc = 0%):
            # Chọn các mẫu mà Margin vượt trội Random nhiều nhất theo Cell Accuracy
            diff_scores = []
            for idx in range(len(test_inputs)):
                m_c = np.mean(m_preds[idx] == test_targets[idx])
                r_c = np.mean(r_preds[idx] == test_targets[idx])
                diff_scores.append((m_c - r_c, m_c, idx))
            diff_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
            chosen_indices = [x[2] for x in diff_scores[:5]]
            print(f"[*] Model pilot đang trong giai đoạn hội tụ: Lọc 5 mẫu tiêu biểu có độ chính xác Margin vượt trội Random nhất.")

        for idx in chosen_indices:
            gold_samples.append({
                "sample_idx": int(idx),
                "quiz_1d": test_inputs[idx],
                "target_1d": test_targets[idx],
                "margin_pred": m_preds[idx],
                "top_prob_pred": t_preds[idx],
                "random_pred": r_preds[idx],
                "margin_history": m_hists[idx],
            })

    gold_file = os.path.join(results_dir, "gold_demo_samples.pkl")
    with open(gold_file, "wb") as f:
        pickle.dump(gold_samples, f)
    print(f"[OK] Đã lưu File 2: {gold_file} ({len(gold_samples)} mẫu)")

    print("=" * 70)
    print("HOÀN TẤT THỰC NGHIỆM TASK 15!")
    print("=" * 70)

    return mdm_results, gold_samples


# ---------------------------------------------------------------------------
# CLI & Unit Test tự động (Kiểm thử Ngày 1-2 & Chạy Task 15)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Inference for Masked Diffusion Models")
    parser.add_argument("--shared_dir", type=str, default=".", help="Thư mục gốc chứa data/, checkpoints/, results/")
    parser.add_argument("--run_experiment", action="store_true", help="Chạy thực nghiệm Task 15")
    parser.add_argument("--quick_test", action="store_true", help="Chạy nhanh 1 seed (42) trên 20 mẫu")
    parser.add_argument("--unit_test_only", action="store_true", help="Chỉ chạy self-test Ngày 1-2")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị ('cpu' hoặc 'cuda')")
    args = parser.parse_args()

    # Nếu được yêu cầu chạy experiment hoặc có file pilot và không cấm
    if args.run_experiment:
        run_task15_experiment(
            shared_dir=args.shared_dir,
            quick_test=args.quick_test,
            device=args.device,
        )
    else:
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
        print("TẤT CẢ UNIT TESTS CỦA TV3 ĐÃ PASS 100%!")
        print("=" * 70)
