"""
src/metrics.py – TV4: Đánh giá & Metrics cho Masked Diffusion Models trên Sudoku
================================================================================
File thuộc quyền quản lý của Thành viên 4 (TV4).

Nhiệm vụ chính:
  1. count_sudoku_violations: Đếm số lượng vi phạm ràng buộc Sudoku (hàng, cột, khối con).
  2. is_valid_sudoku: Kiểm tra nhanh xem bảng Sudoku đã giải đúng và hợp lệ 100% hay chưa.
  3. evaluate_predictions: Đánh giá tổng hợp toàn diện các độ đo (metrics):
     - Puzzle Accuracy (Bảng đúng tuyệt đối so với đáp án).
     - Cell Accuracy (Tỷ lệ từng ô đúng so với đáp án).
     - Valid Puzzle Rate (Tỷ lệ bảng hợp lệ luật Sudoku, 0 vi phạm).
     - Violation Rate (Tỷ lệ bảng có ít nhất một lỗi vi phạm).
     - Mean / Total Violations (Số lỗi vi phạm trung bình / tổng).
     - Mean Steps (Số bước mở ô trung bình qua step_history).
     - Mean Margins (Độ tự tin trung bình của các ô unmasked).
  4. print_evaluation_report: Xuất bảng báo cáo đánh giá đẹp mắt, trực quan.
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
from typing import List, Dict, Tuple, Optional, Any


# Kích thước khối con mặc định theo n_size
BOX_DIMS = {
    4: (2, 2),   # 4x4 : box 2x2
    6: (2, 3),   # 6x6 : box 2x3
    9: (3, 3),   # 9x9 : box 3x3
}


def _resolve_box_dims(n_size: int, box_r: int, box_c: int) -> Tuple[int, int]:
    """Tự động suy luận kích thước khối con (box_r, box_c) phù hợp với n_size."""
    if box_r * box_c == n_size:
        return box_r, box_c
    if n_size in BOX_DIMS:
        return BOX_DIMS[n_size]
    root = int(np.sqrt(n_size))
    if root * root == n_size:
        return root, root
    return box_r, box_c


def count_sudoku_violations(
    board_1d: np.ndarray,
    n_size: int = 9,
    box_r: int = 3,
    box_c: int = 3,
) -> int:
    """Đếm số vi phạm ràng buộc Sudoku trên một bảng.

    Một vi phạm được tính khi một chữ số (từ 1..N) xuất hiện nhiều hơn 1 lần
    trong cùng một hàng, một cột, hoặc một khối con (box).
    Ô trống (0 = [MASK]) không được tính là trùng lặp/vi phạm.

    Parameters
    ----------
    board_1d : np.ndarray, shape (L,), dtype int64
        Bảng Sudoku 1D (có thể chấp nhận mảng 2D (n_size, n_size)).
    n_size : int
        Kích thước bảng (mặc định 9).
    box_r : int
        Số hàng trong mỗi box con.
    box_c : int
        Số cột trong mỗi box con.

    Returns
    -------
    violations : int
        Tổng số vi phạm (trùng lặp) trong hàng, cột, box.
    """
    if board_1d is None:
        return 0

    board_arr = np.asarray(board_1d).reshape(-1)
    if len(board_arr) == 0:
        return 0

    # Tự động đồng bộ n_size và box_r, box_c nếu độ dài phù hợp
    if len(board_arr) != n_size * n_size:
        inferred_n = int(round(np.sqrt(len(board_arr))))
        if inferred_n * inferred_n == len(board_arr):
            n_size = inferred_n

    box_r, box_c = _resolve_box_dims(n_size, box_r, box_c)
    board_2d = board_arr.reshape((n_size, n_size))

    total_violations = 0

    # 1. Ràng buộc hàng (Row constraints)
    for r in range(n_size):
        row_vals = [int(val) for val in board_2d[r, :] if val > 0]
        if len(row_vals) > 0:
            total_violations += len(row_vals) - len(set(row_vals))

    # 2. Ràng buộc cột (Column constraints)
    for c in range(n_size):
        col_vals = [int(val) for val in board_2d[:, c] if val > 0]
        if len(col_vals) > 0:
            total_violations += len(col_vals) - len(set(col_vals))

    # 3. Ràng buộc khối con (Box / Subgrid constraints)
    for br in range(0, n_size, box_r):
        for bc in range(0, n_size, box_c):
            box_vals = [
                int(val)
                for val in board_2d[br : br + box_r, bc : bc + box_c].flatten()
                if val > 0
            ]
            if len(box_vals) > 0:
                total_violations += len(box_vals) - len(set(box_vals))

    return int(total_violations)


def is_valid_sudoku(
    board_1d: np.ndarray,
    n_size: int = 9,
    box_r: int = 3,
    box_c: int = 3,
) -> bool:
    """Kiểm tra xem bảng Sudoku đã được giải hoàn tất và hợp lệ 100% hay chưa.

    Bảng được coi là hợp lệ khi:
      1. Không còn ô trống nào (không có giá trị <= 0 / [MASK]).
      2. Mọi giá trị nằm trong phạm vi hợp lệ [1, n_size].
      3. Không có bất kỳ vi phạm trùng lặp nào ở hàng, cột, khối con (violations == 0).

    Parameters
    ----------
    board_1d : np.ndarray, shape (L,)
        Bảng Sudoku 1D.
    n_size : int
        Kích thước bảng.
    box_r : int
        Số hàng mỗi box.
    box_c : int
        Số cột mỗi box.

    Returns
    -------
    is_valid : bool
        True nếu bảng hoàn chỉnh và hợp lệ, ngược lại False.
    """
    if board_1d is None:
        return False
    board_arr = np.asarray(board_1d).reshape(-1)
    if len(board_arr) == 0:
        return False
    # Không còn ô MASK (0) và không có giá trị ngoài [1, n_size]
    if np.any(board_arr <= 0) or np.any(board_arr > n_size):
        return False
    return count_sudoku_violations(board_arr, n_size, box_r, box_c) == 0


def evaluate_predictions(
    preds_2d: np.ndarray,
    targets_2d: np.ndarray,
    histories: Optional[List[List[Dict]]] = None,
    n_size: int = 9,
    box_r: int = 3,
    box_c: int = 3,
) -> Dict[str, Any]:
    """Đánh giá tổng hợp kết quả dự đoán Sudoku.

    Parameters
    ----------
    preds_2d : np.ndarray, shape (N, L), dtype int64
        Mảng các bảng dự đoán.
    targets_2d : np.ndarray, shape (N, L), dtype int64
        Mảng các bảng đáp án.
    histories : List[List[Dict]], optional
        Danh sách step_history tương ứng mỗi bảng.
    n_size : int
        Kích thước bảng (4, 6, hoặc 9).
    box_r : int
        Số hàng trong mỗi box con.
    box_c : int
        Số cột trong mỗi box con.

    Returns
    -------
    dict_metrics : Dict
        Dictionary chứa các metric đánh giá tổng hợp:
          - 'accuracy'          : float – Tỷ lệ bảng đúng 100% so với targets (alias chuẩn).
          - 'puzzle_accuracy'   : float – Tỷ lệ bảng đúng 100% so với targets.
          - 'cell_accuracy'     : float – Tỷ lệ từng ô dự đoán đúng so với targets.
          - 'valid_puzzle_rate' : float – Tỷ lệ bảng thỏa mãn mọi ràng buộc Sudoku (0 lỗi & không mask).
          - 'violation_rate'    : float – Tỷ lệ bảng bị vi phạm ít nhất 1 ràng buộc.
          - 'mean_violations'   : float – Số lỗi vi phạm ràng buộc trung bình trên mỗi bảng.
          - 'total_violations'  : int   – Tổng số lỗi vi phạm trên toàn bộ tập dữ liệu.
          - 'mean_steps'        : float – Số bước giải trung bình qua các bảng.
          - 'mean_margin'       : float – Độ tự tin (margin) trung bình của các bước unmask.
          - 'num_samples'       : int   – Tổng số lượng mẫu được đánh giá.
    """
    preds_arr = np.asarray(preds_2d)
    targets_arr = np.asarray(targets_2d) if targets_2d is not None else None

    if preds_arr.ndim == 1:
        preds_arr = preds_arr.reshape(1, -1)
    if targets_arr is not None and targets_arr.ndim == 1:
        targets_arr = targets_arr.reshape(1, -1)

    num_samples, seq_len = preds_arr.shape
    box_r, box_c = _resolve_box_dims(n_size, box_r, box_c)

    # 1. Đo lường Accuracy so với Ground Truth Targets
    if targets_arr is not None and targets_arr.shape == preds_arr.shape and num_samples > 0:
        cell_matches = (preds_arr == targets_arr)
        cell_accuracy = float(np.mean(cell_matches))
        puzzle_matches = np.all(cell_matches, axis=1)
        puzzle_accuracy = float(np.mean(puzzle_matches))
    else:
        cell_accuracy = 0.0
        puzzle_accuracy = 0.0

    # 2. Đo lường Violations trên từng bảng
    violations_per_board = [
        count_sudoku_violations(preds_arr[i], n_size=n_size, box_r=box_r, box_c=box_c)
        for i in range(num_samples)
    ]

    total_violations = int(np.sum(violations_per_board))
    mean_violations = float(np.mean(violations_per_board)) if num_samples > 0 else 0.0
    violation_rate = (
        float(np.mean([1.0 if v > 0 else 0.0 for v in violations_per_board]))
        if num_samples > 0
        else 0.0
    )

    # 3. Valid Puzzle Rate: Không vi phạm ràng buộc VÀ không còn ô MASK (0)
    valid_puzzles = [
        1.0 if (v == 0 and not np.any(preds_arr[i] <= 0)) else 0.0
        for i, v in enumerate(violations_per_board)
    ]
    valid_puzzle_rate = float(np.mean(valid_puzzles)) if num_samples > 0 else 0.0

    # 4. Đo lường Metrics từ step_history (Số bước giải, Mean Margin)
    mean_steps = 0.0
    mean_margin = 0.0

    if histories is not None and len(histories) > 0:
        step_counts = []
        all_margins = []
        for h in histories:
            if h and isinstance(h, list):
                step_counts.append(len(h))
                for step_item in h:
                    if isinstance(step_item, dict) and "margins" in step_item:
                        m_list = step_item["margins"]
                        if isinstance(m_list, list) and len(m_list) > 0:
                            all_margins.extend([float(m) for m in m_list])

        if step_counts:
            mean_steps = float(np.mean(step_counts))
        if all_margins:
            mean_margin = float(np.mean(all_margins))

    return {
        "accuracy": puzzle_accuracy,
        "puzzle_accuracy": puzzle_accuracy,
        "cell_accuracy": cell_accuracy,
        "valid_puzzle_rate": valid_puzzle_rate,
        "violation_rate": violation_rate,
        "mean_violations": mean_violations,
        "total_violations": total_violations,
        "mean_steps": mean_steps,
        "mean_margin": mean_margin,
        "num_samples": int(num_samples),
    }


def print_evaluation_report(
    metrics: Dict[str, Any],
    title: str = "BÁO CÁO ĐÁNH GIÁ KẾT QUẢ SUDOKU (TV4)",
) -> None:
    """In bảng tổng hợp các chỉ số đánh giá theo định dạng đẹp mắt."""
    print("=" * 68)
    print(f"📊 {title.upper()}")
    print("=" * 68)
    print(f"  • Số lượng mẫu đánh giá   (num_samples)       : {metrics.get('num_samples', 0):,}")
    print(f"  • Độ chính xác cả bảng    (puzzle_accuracy)   : {metrics.get('puzzle_accuracy', 0.0) * 100:.2f}%")
    print(f"  • Độ chính xác từng ô     (cell_accuracy)     : {metrics.get('cell_accuracy', 0.0) * 100:.2f}%")
    print(f"  • Tỷ lệ bảng hợp lệ luật  (valid_puzzle_rate) : {metrics.get('valid_puzzle_rate', 0.0) * 100:.2f}%")
    print(f"  • Tỷ lệ bảng bị vi phạm   (violation_rate)    : {metrics.get('violation_rate', 0.0) * 100:.2f}%")
    print(f"  • Lỗi vi phạm TB / bảng   (mean_violations)   : {metrics.get('mean_violations', 0.0):.2f}")
    print(f"  • Tổng số vi phạm         (total_violations)  : {metrics.get('total_violations', 0):,}")
    print(f"  • Số bước suy luận TB     (mean_steps)        : {metrics.get('mean_steps', 0.0):.2f}")
    print(f"  • Độ tự tin (margin) TB   (mean_margin)       : {metrics.get('mean_margin', 0.0):.4f}")
    print("=" * 68)


# ---------------------------------------------------------------------------
# Self-test tự động trên dữ liệu giả lập của TV5 (Nghiệm thu Giai đoạn 2 - TV4)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    # Đảm bảo import được src dù chạy từ thư mục gốc learncode hay bên trong intro-ai-project-group3/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    workspace_root = os.path.dirname(project_root)

    for p in (project_root, current_dir, workspace_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from src.interfaces import generate_mock_data, generate_mock_step_history
    except Exception:
        try:
            from interfaces import generate_mock_data, generate_mock_step_history
        except Exception:
            # Fallback độc lập nếu môi trường chưa cài PyTorch hoặc không import được interfaces
            def generate_mock_data(num_samples: int = 10, n_size: int = 9):
                L = n_size * n_size
                solutions = np.random.randint(1, n_size + 1, size=(num_samples, L)).astype(np.int64)
                quizzes = solutions.copy()
                for i in range(num_samples):
                    num_masks = np.random.randint(int(L * 0.4), int(L * 0.7) + 1)
                    mask_indices = np.random.choice(L, size=num_masks, replace=False)
                    quizzes[i, mask_indices] = 0
                return quizzes, solutions

            def generate_mock_step_history(quiz_1d: np.ndarray, n_size: int = 9):
                L = len(quiz_1d)
                board = quiz_1d.copy().astype(np.int64)
                mask_positions = np.where(board == 0)[0].tolist()
                np.random.shuffle(mask_positions)
                step_history = []
                for step_idx, pos in enumerate(mask_positions, start=1):
                    value = int(np.random.randint(1, n_size + 1))
                    margin = float(np.random.uniform(0.0, 1.0))
                    board[pos] = value
                    step_history.append({
                        "step": step_idx,
                        "unmasked_indices": [int(pos)],
                        "values": [value],
                        "margins": [margin],
                        "board_state": board.copy(),
                    })
                return step_history

    print("=" * 68)
    print("KIỂM THỬ MODULE METRICS (TV4) - SẴN SÀNG BỘ CHẤM ĐIỂM TỰ ĐỘNG")
    print("=" * 68)

    # -----------------------------------------------------------------------
    # TEST 1: count_sudoku_violations trên bảng 4x4 hợp lệ (0 vi phạm)
    # -----------------------------------------------------------------------
    print("\n[TEST 1] Kiểm tra bảng Sudoku 4x4 hợp lệ (0 vi phạm)...")
    valid_4x4 = np.array([
        1, 2, 3, 4,
        3, 4, 1, 2,
        2, 1, 4, 3,
        4, 3, 2, 1,
    ], dtype=np.int64)

    v_valid = count_sudoku_violations(valid_4x4, n_size=4, box_r=2, box_c=2)
    assert v_valid == 0, f"Bảng hợp lệ nhưng trả về {v_valid} vi phạm!"
    assert is_valid_sudoku(valid_4x4, n_size=4, box_r=2, box_c=2), "is_valid_sudoku phải trả về True"
    print("  ✅ [PASS] count_sudoku_violations = 0 và is_valid_sudoku = True.")

    # -----------------------------------------------------------------------
    # TEST 2: count_sudoku_violations trên bảng 4x4 có vi phạm cố tình
    # -----------------------------------------------------------------------
    print("\n[TEST 2] Kiểm tra bảng Sudoku 4x4 có vi phạm có chủ đích...")
    # Thay ô (0, 1) từ 2 thành 1 -> Hàng 0 có [1, 1, 3, 4] (1 vi phạm hàng)
    # Cột 1 có [1, 4, 1, 3] -> (1 vi phạm cột)
    # Box (0,0) có [1, 1, 3, 4] -> (1 vi phạm box)
    # Tổng cộng = 3 vi phạm
    invalid_4x4 = valid_4x4.copy()
    invalid_4x4[1] = 1  # vị trí (0, 1)

    v_invalid = count_sudoku_violations(invalid_4x4, n_size=4, box_r=2, box_c=2)
    assert v_invalid == 3, f"Kỳ vọng 3 vi phạm nhưng nhận được {v_invalid}!"
    assert not is_valid_sudoku(invalid_4x4, n_size=4, box_r=2, box_c=2), "is_valid_sudoku phải trả về False"
    print(f"  ✅ [PASS] Phát hiện chính xác {v_invalid} vi phạm (hàng, cột, khối con).")

    # -----------------------------------------------------------------------
    # TEST 3: Ô trống (0 = [MASK]) không được tính là lỗi trùng lặp
    # -----------------------------------------------------------------------
    print("\n[TEST 3] Kiểm tra bảng dở dang chứa nhiều ô [MASK] (0)...")
    partial_4x4 = np.array([
        1, 0, 0, 4,
        0, 4, 1, 0,
        2, 0, 0, 3,
        0, 3, 2, 0,
    ], dtype=np.int64)
    v_partial = count_sudoku_violations(partial_4x4, n_size=4, box_r=2, box_c=2)
    assert v_partial == 0, f"Các ô [MASK] bị tính nhầm là trùng lặp! vi phạm = {v_partial}"
    assert not is_valid_sudoku(partial_4x4, n_size=4, box_r=2, box_c=2), "Bảng còn ô [MASK] không được coi là hoàn tất!"
    print("  ✅ [PASS] Ô [MASK] (0) không bị đếm trùng lặp.")

    # -----------------------------------------------------------------------
    # TEST 4: evaluate_predictions khi đáp án đúng 100% (Perfect Match)
    # -----------------------------------------------------------------------
    print("\n[TEST 4] Kiểm tra evaluate_predictions khi dự đoán hoàn hảo (100%)...")
    perfect_preds = np.tile(valid_4x4, (5, 1))
    perfect_targets = np.tile(valid_4x4, (5, 1))
    perfect_metrics = evaluate_predictions(
        preds_2d=perfect_preds,
        targets_2d=perfect_targets,
        histories=None,
        n_size=4,
        box_r=2,
        box_c=2,
    )
    assert perfect_metrics["puzzle_accuracy"] == 1.0, "Puzzle accuracy phải là 1.0"
    assert perfect_metrics["cell_accuracy"] == 1.0, "Cell accuracy phải là 1.0"
    assert perfect_metrics["valid_puzzle_rate"] == 1.0, "Valid puzzle rate phải là 1.0"
    assert perfect_metrics["total_violations"] == 0, "Total violations phải là 0"
    print("  ✅ [PASS] Perfect match: Accuracy = 100%, Violations = 0.")

    # -----------------------------------------------------------------------
    # TEST 5: evaluate_predictions trên Mock Data của TV5 (9x9)
    # -----------------------------------------------------------------------
    print("\n[TEST 5] Kiểm tra evaluate_predictions trên dữ liệu giả lập của TV5 (9x9)...")
    quizzes, solutions = generate_mock_data(num_samples=10, n_size=9)
    # Giả lập histories từ TV5
    mock_histories = [generate_mock_step_history(quizzes[i], n_size=9) for i in range(10)]
    # Bảng dự đoán giả lập: lấy cuối step_history của mỗi bảng
    mock_preds = np.array([hist[-1]["board_state"] for hist in mock_histories])

    mock_metrics = evaluate_predictions(
        preds_2d=mock_preds,
        targets_2d=solutions,
        histories=mock_histories,
        n_size=9,
        box_r=3,
        box_c=3,
    )

    # Kiểm tra đầy đủ các key chuẩn
    REQUIRED_METRICS = {
        "accuracy", "puzzle_accuracy", "cell_accuracy", "valid_puzzle_rate",
        "violation_rate", "mean_violations", "total_violations",
        "mean_steps", "mean_margin", "num_samples",
    }
    assert REQUIRED_METRICS.issubset(set(mock_metrics.keys())), "Thiếu các metric bắt buộc!"
    assert mock_metrics["num_samples"] == 10, f"Num samples sai: {mock_metrics['num_samples']}"
    assert mock_metrics["mean_steps"] > 0, "Mean steps phải > 0"
    assert 0.0 <= mock_metrics["mean_margin"] <= 1.0, "Mean margin phải trong [0, 1]"
    print("  ✅ [PASS] Đánh giá thành công trên 10 bảng 9x9 với đầy đủ các metrics.")

    # In báo cáo tổng hợp nghiệm thu
    print("\n")
    print_evaluation_report(mock_metrics, title="Báo Cáo Nghiệm Thu TV4 (Mock Data)")

    print("\n" + "=" * 68)
    print("🎉 TẤT CẢ UNIT TESTS CỦA TV4 ĐÃ PASS 100%! BỘ CHẤM ĐIỂM SẴN SÀNG.")
    print("=" * 68)
