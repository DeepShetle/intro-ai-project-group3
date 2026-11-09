"""
src/metrics.py – TV4: Đánh giá & Metrics cho Masked Diffusion Models trên Sudoku
================================================================================
File thuộc quyền quản lý của Thành viên 4 (TV4).

Nhiệm vụ chính:
  Ngày 1:
    1. count_sudoku_violations: Đếm số lượng vi phạm ràng buộc Sudoku (hàng, cột, khối con).
    2. is_valid_sudoku: Kiểm tra nhanh xem bảng Sudoku đã giải đúng và hợp lệ 100% hay chưa.
    3. evaluate_predictions: Đánh giá tổng hợp toàn diện các độ đo (metrics):
       - board_accuracy_pct (và puzzle_accuracy): Tỷ lệ bảng đúng 100% so với đáp án.
       - cell_accuracy_pct (và cell_accuracy): Tỷ lệ từng ô đúng so với đáp án.
       - avg_constraint_violations (và mean_violations): Số lỗi vi phạm trung bình / bảng.
       - high_margin_cell_acc_pct: Tỷ lệ chính xác của các ô có margin >= threshold_high (0.5).
       - low_margin_cell_acc_pct: Tỷ lệ chính xác của các ô có margin < threshold_low (0.2 hoặc 0.5).
       - valid_puzzle_rate, violation_rate, mean_steps, mean_margin.
    4. print_evaluation_report: Xuất bảng báo cáo đánh giá đẹp mắt, trực quan.

  Ngày 2:
    5. analyze_margin_vs_accuracy: Phân tích tương quan giữa Margin Confidence và độ chính xác từng ô.
    6. aggregate_multi_seed_results: Tổng hợp mean và std qua nhiều random seeds (ví dụ 3 seeds).
    7. plot_strategy_comparison: Vẽ biểu đồ so sánh các chiến lược suy luận (random, top_prob, margin, csp),
       xuất file PNG / PDF chất lượng cao.
    8. plot_margin_vs_accuracy: Vẽ biểu đồ Margin Confidence vs Cell Accuracy (%), xuất file PNG / PDF.
"""

from __future__ import annotations

import os
import sys
import glob

# Tự động cấu hình môi trường hiển thị và nạp thư viện
try:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Patch hỗ trợ macOS sandbox và uv cache nếu cần
if "/tmp/pyexpat_fix" not in sys.path and os.path.exists("/tmp/pyexpat_fix"):
    sys.path.insert(0, "/tmp/pyexpat_fix")
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

_uv_archives = glob.glob(os.path.expanduser("~/.cache/uv/archive-v0/*"))
for _d in _uv_archives:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union

# Import matplotlib an toàn cho môi trường headless
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    matplotlib = None
    plt = None
    _HAS_MATPLOTLIB = False


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


# ===========================================================================
# 1. Metric Summary Wrapper hỗ trợ cả dạng số, dict và truy cập thuộc tính
# ===========================================================================

class MetricSummary(float):
    """Wrapper cho giá trị metric tổng hợp qua nhiều seed: vừa là float (mean),
    vừa hỗ trợ truy cập ['mean'], ['std'], .mean, .std và in định dạng mean ± std.
    """

    def __new__(cls, mean: float, std: float = 0.0, raw: Optional[List[float]] = None):
        obj = super().__new__(cls, float(mean))
        obj.mean = float(mean)
        obj.std = float(std)
        obj.raw = raw or []
        return obj

    def __getitem__(self, item: str) -> Any:
        if item == "mean":
            return self.mean
        elif item == "std":
            return self.std
        elif item in ("raw", "values"):
            return self.raw
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        if item == "mean":
            return self.mean
        elif item == "std":
            return self.std
        elif item in ("raw", "values"):
            return self.raw
        return default

    def __repr__(self) -> str:
        return f"{self.mean:.4f} ± {self.std:.4f}"

    def __str__(self) -> str:
        return f"{self.mean:.2f} ± {self.std:.2f}"

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}


# ===========================================================================
# 2. Các hàm kiểm tra và đánh giá cơ bản (Ngày 1)
# ===========================================================================

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
    threshold_high: float = 0.5,
    threshold_low: float = 0.2,
) -> Dict[str, Any]:
    """Đánh giá tổng hợp toàn diện kết quả dự đoán Sudoku.

    Parameters
    ----------
    preds_2d : np.ndarray, shape (N, L), dtype int64
        Mảng các bảng dự đoán.
    targets_2d : np.ndarray, shape (N, L), dtype int64
        Mảng các bảng đáp án chuẩn (ground truth).
    histories : List[List[Dict]], optional
        Danh sách step_history tương ứng mỗi bảng.
    n_size : int
        Kích thước bảng (4, 6, hoặc 9).
    box_r : int
        Số hàng trong mỗi box con.
    box_c : int
        Số cột trong mỗi box con.
    threshold_high : float
        Ngưỡng margin cao (mặc định 0.5).
    threshold_low : float
        Ngưỡng margin thấp (mặc định 0.2, có thể chọn 0.5 tùy cấu hình nghiên cứu).

    Returns
    -------
    dict_metrics : Dict[str, Any]
        Dictionary chứa các keys chuẩn hóa và tương thích:
          - 'board_accuracy_pct'       : float – Tỷ lệ bảng đúng 100% (theo %).
          - 'cell_accuracy_pct'        : float – Tỷ lệ từng ô dự đoán đúng (theo %).
          - 'avg_constraint_violations': float – Số lỗi vi phạm trung bình / bảng.
          - 'high_margin_cell_acc_pct' : float – Độ chính xác ô có margin >= threshold_high (%).
          - 'low_margin_cell_acc_pct'  : float – Độ chính xác ô có margin < threshold_low (%).
          - 'puzzle_accuracy'          : float – Tỷ lệ bảng đúng (tỷ lệ 0.0 - 1.0).
          - 'cell_accuracy'            : float – Tỷ lệ ô đúng (tỷ lệ 0.0 - 1.0).
          - 'mean_violations'          : float – Alias của avg_constraint_violations.
          - 'total_violations'         : int   – Tổng số lỗi vi phạm.
          - 'valid_puzzle_rate'        : float – Tỷ lệ bảng hợp lệ hoàn toàn (0 lỗi & không mask).
          - 'violation_rate'           : float – Tỷ lệ bảng có >= 1 lỗi vi phạm.
          - 'mean_steps'               : float – Số bước giải trung bình.
          - 'mean_margin'              : float – Margin trung bình của các ô unmask.
          - 'num_samples'              : int   – Tổng số mẫu đánh giá.
    """
    preds_arr = np.asarray(preds_2d)
    targets_arr = np.asarray(targets_2d) if targets_2d is not None else None

    if preds_arr.ndim == 1:
        preds_arr = preds_arr.reshape(1, -1)
    if targets_arr is not None and targets_arr.ndim == 1:
        targets_arr = targets_arr.reshape(1, -1)

    num_samples = len(preds_arr)
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

    board_accuracy_pct = float(puzzle_accuracy * 100.0)
    cell_accuracy_pct = float(cell_accuracy * 100.0)

    # 2. Đo lường Violations trên từng bảng
    violations_per_board = [
        count_sudoku_violations(preds_arr[i], n_size=n_size, box_r=box_r, box_c=box_c)
        for i in range(num_samples)
    ]

    total_violations = int(np.sum(violations_per_board))
    mean_violations = float(np.mean(violations_per_board)) if num_samples > 0 else 0.0
    avg_constraint_violations = mean_violations
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

    # 4. Đo lường Metrics từ step_history (Số bước, Mean Margin, Margin Accuracy)
    mean_steps = 0.0
    mean_margin = 0.0
    high_margin_cell_acc_pct = 0.0
    low_margin_cell_acc_pct = 0.0
    high_margin_total = 0
    low_margin_total = 0

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

        # Phân tích chi tiết high_margin vs low_margin nếu có targets_arr
        if targets_arr is not None:
            margin_analysis = analyze_margin_vs_accuracy(
                step_histories=histories,
                targets_2d=targets_arr,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
            )
            high_margin_cell_acc_pct = margin_analysis["high_margin_cell_acc_pct"]
            low_margin_cell_acc_pct = margin_analysis["low_margin_cell_acc_pct"]
            high_margin_total = margin_analysis["high_margin_total"]
            low_margin_total = margin_analysis["low_margin_total"]

    return {
        # --- Các keys chuẩn hóa theo yêu cầu ---
        "board_accuracy_pct": board_accuracy_pct,
        "cell_accuracy_pct": cell_accuracy_pct,
        "avg_constraint_violations": avg_constraint_violations,
        "high_margin_cell_acc_pct": high_margin_cell_acc_pct,
        "low_margin_cell_acc_pct": low_margin_cell_acc_pct,

        # --- Các keys chuẩn gốc Ngày 1 để tương thích ngược ---
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

        # --- Bổ sung thông tin số lượng mẫu phân nhóm ---
        "board_accuracy": puzzle_accuracy,
        "high_margin_total": high_margin_total,
        "low_margin_total": low_margin_total,
    }


def print_evaluation_report(
    metrics: Dict[str, Any],
    title: str = "BÁO CÁO ĐÁNH GIÁ KẾT QUẢ SUDOKU (TV4)",
) -> None:
    """In bảng tổng hợp các chỉ số đánh giá theo định dạng trực quan, đẹp mắt."""
    print("=" * 72)
    print(f"📊 {title.upper()}")
    print("=" * 72)
    print(f"  • Số lượng mẫu đánh giá       (num_samples)           : {metrics.get('num_samples', 0):,}")
    print(f"  • Độ chính xác cả bảng        (board_accuracy_pct)   : {metrics.get('board_accuracy_pct', metrics.get('puzzle_accuracy', 0.0) * 100):.2f}%")
    print(f"  • Độ chính xác từng ô         (cell_accuracy_pct)     : {metrics.get('cell_accuracy_pct', metrics.get('cell_accuracy', 0.0) * 100):.2f}%")
    print(f"  • Lỗi vi phạm TB / bảng       (avg_violations)        : {metrics.get('avg_constraint_violations', metrics.get('mean_violations', 0.0)):.2f}")
    print(f"  • Tỷ lệ bảng hợp lệ luật      (valid_puzzle_rate)     : {metrics.get('valid_puzzle_rate', 0.0) * 100:.2f}%")
    print(f"  • Tỷ lệ bảng bị vi phạm       (violation_rate)        : {metrics.get('violation_rate', 0.0) * 100:.2f}%")
    print(f"  • Tổng số lỗi vi phạm         (total_violations)      : {metrics.get('total_violations', 0):,}")
    print(f"  • Số bước suy luận TB         (mean_steps)            : {metrics.get('mean_steps', 0.0):.2f}")
    print(f"  • Độ tự tin (margin) TB       (mean_margin)           : {metrics.get('mean_margin', 0.0):.4f}")
    if "high_margin_cell_acc_pct" in metrics:
        print(f"  • High-Margin Cell Acc (>=0.5)(high_margin_acc_pct)   : {metrics.get('high_margin_cell_acc_pct', 0.0):.2f}%")
        print(f"  • Low-Margin Cell Acc  (<0.2) (low_margin_acc_pct)    : {metrics.get('low_margin_cell_acc_pct', 0.0):.2f}%")
    print("=" * 72)


# ===========================================================================
# 3. Các hàm phân tích & tổng hợp nâng cao (Ngày 2)
# ===========================================================================

def analyze_margin_vs_accuracy(
    step_histories: Union[List[List[Dict]], List[Dict]],
    targets_2d: np.ndarray,
    threshold_high: float = 0.5,
    threshold_low: float = 0.2,
) -> Dict[str, Any]:
    """Phân tích tương quan giữa Margin Confidence và độ chính xác của từng ô unmask.

    So khớp từng vị trí `unmasked_indices` và `values` trong `step_history`
    với `targets_2d` để kiểm tra ô mở ra đúng hay sai.
    Được trang bị cơ chế bảo vệ chia cho 0 (`len > 0 else 0.0`) khi không có ô nào
    thuộc nhóm margin tương ứng.

    Parameters
    ----------
    step_histories : List[List[Dict]] hoặc List[Dict]
        Danh sách step_history của các bảng (hoặc 1 bảng).
    targets_2d : np.ndarray
        Mảng 2D chứa bảng đáp án đúng (shape (N, L)) hoặc 1D (shape (L,)).
    threshold_high : float
        Ngưỡng margin cao (mặc định 0.5).
    threshold_low : float
        Ngưỡng margin thấp (mặc định 0.2).

    Returns
    -------
    analysis : Dict[str, Any]
        Dictionary chứa kết quả phân tích:
          - 'high_margin_cell_acc_pct': float (độ chính xác % nhóm margin >= threshold_high)
          - 'low_margin_cell_acc_pct' : float (độ chính xác % nhóm margin < threshold_low)
          - 'high_margin_total'       : int (tổng số ô nhóm margin cao)
          - 'high_margin_correct'     : int (số ô đúng nhóm margin cao)
          - 'low_margin_total'        : int (tổng số ô nhóm margin thấp)
          - 'low_margin_correct'      : int (số ô đúng nhóm margin thấp)
          - 'medium_margin_cell_acc_pct': float (độ chính xác % nhóm ngưỡng giữa [low, high))
          - 'bins'                    : list[str] (nhãn khoảng margin)
          - 'bin_counts'              : list[int] (số lượng ô trong mỗi khoảng)
          - 'bin_accuracies_pct'      : list[float] (độ chính xác % trong mỗi khoảng)
          - 'total_cells'             : int (tổng số ô unmask được phân tích)
          - 'overall_cell_acc_pct'    : float (độ chính xác chung toàn bộ các ô unmask)
    """
    if targets_2d is None:
        return {
            "high_margin_cell_acc_pct": 0.0,
            "low_margin_cell_acc_pct": 0.0,
            "high_margin_total": 0,
            "high_margin_correct": 0,
            "low_margin_total": 0,
            "low_margin_correct": 0,
            "medium_margin_cell_acc_pct": 0.0,
            "medium_margin_total": 0,
            "medium_margin_correct": 0,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "bins": ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
            "bin_counts": [0, 0, 0, 0, 0],
            "bin_accuracies_pct": [0.0, 0.0, 0.0, 0.0, 0.0],
            "total_cells": 0,
            "overall_cell_acc_pct": 0.0,
            "mean_margin": 0.0,
        }

    targets_arr = np.asarray(targets_2d)
    if targets_arr.ndim == 1:
        targets_arr = targets_arr.reshape(1, -1)

    # Chuẩn hóa step_histories nếu truyền vào 1 bảng đơn lẻ
    if step_histories and isinstance(step_histories, list) and len(step_histories) > 0:
        if isinstance(step_histories[0], dict):
            histories_list = [step_histories]
        else:
            histories_list = step_histories
    else:
        histories_list = []

    records: List[Tuple[float, bool]] = []

    num_samples = min(len(histories_list), len(targets_arr))
    for i in range(num_samples):
        target_row = targets_arr[i]
        sample_hist = histories_list[i]
        if not sample_hist or not isinstance(sample_hist, list):
            continue

        for step_dict in sample_hist:
            if not isinstance(step_dict, dict):
                continue
            unmasked_indices = step_dict.get("unmasked_indices", [])
            values = step_dict.get("values", [])
            margins = step_dict.get("margins", [])

            limit = min(len(unmasked_indices), len(values), len(margins))
            for k in range(limit):
                idx = int(unmasked_indices[k])
                val = int(values[k])
                margin = float(margins[k])
                is_correct = bool(val == target_row[idx])
                records.append((margin, is_correct))

    # Nhóm high margin: margin >= threshold_high
    high_group = [correct for m, correct in records if m >= threshold_high]
    high_total = len(high_group)
    high_correct = sum(high_group)
    high_acc_pct = float(high_correct / high_total * 100.0) if high_total > 0 else 0.0

    # Nhóm low margin: margin < threshold_low
    low_group = [correct for m, correct in records if m < threshold_low]
    low_total = len(low_group)
    low_correct = sum(low_group)
    low_acc_pct = float(low_correct / low_total * 100.0) if low_total > 0 else 0.0

    # Nhóm medium margin: threshold_low <= margin < threshold_high
    medium_group = [correct for m, correct in records if threshold_low <= m < threshold_high]
    medium_total = len(medium_group)
    medium_correct = sum(medium_group)
    medium_acc_pct = float(medium_correct / medium_total * 100.0) if medium_total > 0 else 0.0

    # Phân phối theo 5 bins [0.0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0] phục vụ vẽ đồ thị
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bins = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    bin_counts = []
    bin_accuracies_pct = []

    for b_idx in range(len(bins)):
        low_e = bin_edges[b_idx]
        high_e = bin_edges[b_idx + 1]
        if b_idx == len(bins) - 1:
            in_bin = [c for m, c in records if low_e <= m <= high_e]
        else:
            in_bin = [c for m, c in records if low_e <= m < high_e]

        cnt = len(in_bin)
        corr = sum(in_bin)
        acc_pct = float(corr / cnt * 100.0) if cnt > 0 else 0.0
        bin_counts.append(cnt)
        bin_accuracies_pct.append(acc_pct)

    total_cells = len(records)
    overall_cell_acc_pct = (
        float(sum(c for _, c in records) / total_cells * 100.0) if total_cells > 0 else 0.0
    )
    mean_margin = float(np.mean([m for m, _ in records])) if total_cells > 0 else 0.0

    return {
        "high_margin_cell_acc_pct": high_acc_pct,
        "low_margin_cell_acc_pct": low_acc_pct,
        "high_margin_acc_pct": high_acc_pct,
        "low_margin_acc_pct": low_acc_pct,
        "high_margin_total": high_total,
        "high_margin_correct": high_correct,
        "low_margin_total": low_total,
        "low_margin_correct": low_correct,
        "medium_margin_cell_acc_pct": medium_acc_pct,
        "medium_margin_total": medium_total,
        "medium_margin_correct": medium_correct,
        "threshold_high": threshold_high,
        "threshold_low": threshold_low,
        "bins": bins,
        "bin_counts": bin_counts,
        "bin_accuracies_pct": bin_accuracies_pct,
        "total_cells": total_cells,
        "overall_cell_acc_pct": overall_cell_acc_pct,
        "mean_margin": mean_margin,
    }


def aggregate_multi_seed_results(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tính mean và std cho từng metric qua nhiều random seeds (ví dụ 3 random seeds).

    Hỗ trợ cả:
      - result[metric]["mean"] và result[metric]["std"]
      - result[f"{metric}_mean"] và result[f"{metric}_std"]
      - result["mean"][metric] và result["std"][metric]
      - result[metric] vừa có thể dùng như số float, vừa in ra 'mean ± std'.

    Parameters
    ----------
    seed_results : List[Dict[str, Any]]
        Danh sách kết quả metrics thu được từ các seeds khác nhau.

    Returns
    -------
    aggregated : Dict[str, Any]
        Dictionary chứa giá trị mean và std cho từng metric.
    """
    if not seed_results:
        return {}

    # Thu thập tất cả các keys có giá trị số
    all_keys = set()
    for res in seed_results:
        if isinstance(res, dict):
            all_keys.update(res.keys())

    aggregated: Dict[str, Any] = {}
    mean_dict: Dict[str, float] = {}
    std_dict: Dict[str, float] = {}

    for key in sorted(all_keys):
        vals: List[float] = []
        for res in seed_results:
            if not isinstance(res, dict) or key not in res:
                continue
            val = res[key]
            if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
                vals.append(float(val))

        if vals:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            summary = MetricSummary(mean=mean_val, std=std_val, raw=vals)

            aggregated[key] = summary
            aggregated[f"{key}_mean"] = mean_val
            aggregated[f"{key}_std"] = std_val
            mean_dict[key] = mean_val
            std_dict[key] = std_val

    aggregated["mean"] = mean_dict
    aggregated["std"] = std_dict
    aggregated["num_seeds"] = len(seed_results)

    return aggregated


# ===========================================================================
# 4. Các hàm vẽ biểu đồ xuất file PNG/PDF (Ngày 2)
# ===========================================================================

def _extract_metric_pair(
    data: Dict[str, Any],
    *key_candidates: str,
    as_percentage: bool = False,
) -> Tuple[float, float]:
    """Trích xuất (mean, std) an toàn từ dict kết quả theo danh sách khóa ứng viên."""
    for k in key_candidates:
        if k in data:
            val = data[k]
            if isinstance(val, dict):
                m = float(val.get("mean", 0.0))
                s = float(val.get("std", 0.0))
                return m, s
            if hasattr(val, "mean") and hasattr(val, "std"):
                return float(val.mean), float(val.std)
            if isinstance(val, (int, float, np.number)):
                m = float(val)
                s = float(data.get(f"{k}_std", 0.0))
                if as_percentage and m <= 1.0 and m > 0.0 and not k.endswith("_pct"):
                    m *= 100.0
                    s *= 100.0
                return m, s
    return 0.0, 0.0


def plot_strategy_comparison(
    results_by_strategy: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
) -> Any:
    """Vẽ biểu đồ so sánh các chiến lược giải mã Sudoku (Ngày 2).

    So sánh 3 chỉ số then chốt trên cùng 1 biểu đồ đa khung nhìn:
      1. Board Accuracy (%) – Độ chính xác hoàn chỉnh cả bảng (càng cao càng tốt).
      2. Cell Accuracy (%) – Độ chính xác từng ô (càng cao càng tốt).
      3. Avg Violations – Số lỗi vi phạm ràng buộc Sudoku TB (càng thấp càng tốt).

    Parameters
    ----------
    results_by_strategy : Dict[str, Dict[str, Any]]
        Dictionary ánh xạ tên chiến lược ('random', 'top_prob', 'margin', 'csp'...)
        tới dictionary metrics tương ứng (hoặc kết quả từ aggregate_multi_seed_results).
    save_path : Optional[str]
        Đường dẫn lưu file biểu đồ (.png hoặc .pdf). Mặc định không lưu.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Đối tượng hình vẽ matplotlib.
    """
    if not _HAS_MATPLOTLIB:
        print("⚠️ Cảnh báo: Matplotlib chưa được cài đặt. Không thể vẽ biểu đồ.")
        return None

    if not results_by_strategy:
        print("⚠️ Cảnh báo: results_by_strategy rỗng, bỏ qua vẽ biểu đồ.")
        return None

    strategies = list(results_by_strategy.keys())
    board_accs: List[float] = []
    board_stds: List[float] = []
    cell_accs: List[float] = []
    cell_stds: List[float] = []
    violations: List[float] = []
    viol_stds: List[float] = []

    for strat in strategies:
        data = results_by_strategy[strat]
        b_m, b_s = _extract_metric_pair(data, "board_accuracy_pct", "puzzle_accuracy", as_percentage=True)
        c_m, c_s = _extract_metric_pair(data, "cell_accuracy_pct", "cell_accuracy", as_percentage=True)
        v_m, v_s = _extract_metric_pair(data, "avg_constraint_violations", "mean_violations")

        board_accs.append(b_m)
        board_stds.append(b_s)
        cell_accs.append(c_m)
        cell_stds.append(c_s)
        violations.append(v_m)
        viol_stds.append(v_s)

    # Bảng màu hài hòa theo nhận diện nhóm
    color_palette = ["#7f8c8d", "#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    colors = [color_palette[i % len(color_palette)] for i in range(len(strategies))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=100)

    # 1. Panel: Board Accuracy (%)
    ax0 = axes[0]
    bars0 = ax0.bar(
        strategies,
        board_accs,
        yerr=board_stds if any(s > 0 for s in board_stds) else None,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    ax0.set_title("Board Accuracy (%)\n(Càng cao càng tốt ↑)", fontsize=12, fontweight="bold")
    ax0.set_ylabel("Độ chính xác (%)", fontsize=10)
    ax0.set_ylim(0, max(105, max(board_accs, default=0) + 15))
    ax0.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars0:
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Panel: Cell Accuracy (%)
    ax1 = axes[1]
    bars1 = ax1.bar(
        strategies,
        cell_accs,
        yerr=cell_stds if any(s > 0 for s in cell_stds) else None,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    ax1.set_title("Cell Accuracy (%)\n(Càng cao càng tốt ↑)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Độ chính xác (%)", fontsize=10)
    ax1.set_ylim(0, max(105, max(cell_accs, default=0) + 15))
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Panel: Avg Violations
    ax2 = axes[2]
    bars2 = ax2.bar(
        strategies,
        violations,
        yerr=viol_stds if any(s > 0 for s in viol_stds) else None,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    ax2.set_title("Lỗi vi phạm TB / Bảng\n(Càng thấp càng tốt ↓)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Số lỗi vi phạm", fontsize=10)
    ax2.set_ylim(0, max(5, max(violations, default=0) * 1.25 + 0.5))
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.suptitle("SO SÁNH CÁC CHIẾN LƯỢC SUY LUẬN THÍCH ỨNG SUDOKU (TV4)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  💾 [PNG/PDF] Đã xuất biểu đồ so sánh vào: {save_path}")

    return fig


def plot_margin_vs_accuracy(
    margin_analysis: Dict[str, Any],
    save_path: Optional[str] = None,
) -> Any:
    """Vẽ biểu đồ tương quan Margin Confidence vs Cell Accuracy (%) (Ngày 2).

    Minh họa trực quan rằng độ tự tin của mô hình (Margin) tương quan thuận
    với độ chính xác của từng ô unmasked, đúng như giả thuyết đề tài ICML 2025.

    Parameters
    ----------
    margin_analysis : Dict[str, Any]
        Dictionary trả về từ hàm analyze_margin_vs_accuracy.
    save_path : Optional[str]
        Đường dẫn lưu file biểu đồ (.png hoặc .pdf). Mặc định không lưu.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Đối tượng hình vẽ matplotlib.
    """
    if not _HAS_MATPLOTLIB:
        print("⚠️ Cảnh báo: Matplotlib chưa được cài đặt. Không thể vẽ biểu đồ.")
        return None

    if not margin_analysis:
        print("⚠️ Cảnh báo: margin_analysis rỗng, bỏ qua vẽ biểu đồ.")
        return None

    bins = margin_analysis.get("bins", ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"])
    accs = margin_analysis.get("bin_accuracies_pct", [0.0] * len(bins))
    counts = margin_analysis.get("bin_counts", [0] * len(bins))

    high_acc = float(margin_analysis.get("high_margin_cell_acc_pct", 0.0))
    low_acc = float(margin_analysis.get("low_margin_cell_acc_pct", 0.0))
    th_high = float(margin_analysis.get("threshold_high", 0.5))
    th_low = float(margin_analysis.get("threshold_low", 0.2))

    fig, (ax_main, ax_compare) = plt.subplots(1, 2, figsize=(13, 5), dpi=100, gridspec_kw={"width_ratios": [2.2, 1]})

    # 1. Trục chính: Bins độ tự tin vs Độ chính xác
    bar_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    bars = ax_main.bar(
        bins,
        accs,
        color=bar_colors[: len(bins)],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    ax_main.set_title("Độ chính xác từng ô theo dải Margin Confidence", fontsize=12, fontweight="bold")
    ax_main.set_xlabel("Khoảng giá trị Margin (Top 1 - Top 2)", fontsize=10)
    ax_main.set_ylabel("Độ chính xác (%)", fontsize=10)
    ax_main.set_ylim(0, 105)
    ax_main.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, cnt in zip(bars, counts):
        h = bar.get_height()
        ax_main.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.5,
            f"{h:.1f}%\n(n={cnt})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # 2. Trục so sánh đối đầu: Low Margin vs High Margin
    labels = [f"Low Margin\n(< {th_low})", f"High Margin\n(≥ {th_high})"]
    compare_vals = [low_acc, high_acc]
    compare_colors = ["#e67e22", "#27ae60"]

    comp_bars = ax_compare.bar(
        labels,
        compare_vals,
        color=compare_colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
        width=0.55,
    )
    ax_compare.set_title("So sánh trực tiếp\nLow vs High Margin", fontsize=12, fontweight="bold")
    ax_compare.set_ylabel("Độ chính xác (%)", fontsize=10)
    ax_compare.set_ylim(0, 105)
    ax_compare.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in comp_bars:
        h = bar.get_height()
        ax_compare.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 2.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.suptitle("PHÂN TÍCH TƯƠNG QUAN MARGIN CONFIDENCE VS CELL ACCURACY (TV4)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  💾 [PNG/PDF] Đã xuất biểu đồ phân tích margin vào: {save_path}")

    return fig


# ===========================================================================
# 5. Self-test tự động toàn diện kiểm thử Ngày 1 & Ngày 2
# ===========================================================================

if __name__ == "__main__":
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

    print("=" * 72)
    print("KIỂM THỬ TOÀN DIỆN MODULE METRICS (TV4) - SẴN SÀNG BỘ ĐÁNH GIÁ NGÀY 1 & 2")
    print("=" * 72)

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
    invalid_4x4 = valid_4x4.copy()
    invalid_4x4[1] = 1  # vị trí (0, 1) gây 3 vi phạm: hàng 0, cột 1, box (0,0)

    v_invalid = count_sudoku_violations(invalid_4x4, n_size=4, box_r=2, box_c=2)
    assert v_invalid == 3, f"Kỳ vọng 3 vi phạm nhưng nhận được {v_invalid}!"
    assert not is_valid_sudoku(invalid_4x4, n_size=4, box_r=2, box_c=2), "is_valid_sudoku phải trả về False"
    print(f"  ✅ [PASS] Phát hiện chính xác {v_invalid} vi phạm.")

    # -----------------------------------------------------------------------
    # TEST 3: Ô trống (0 = [MASK]) không được tính là lỗi trùng lặp
    # -----------------------------------------------------------------------
    print("\n[TEST 3] Kiểm tra ô trống [MASK] (0) không bị đếm trùng lặp...")
    partial_4x4 = np.array([
        1, 0, 0, 4,
        0, 4, 1, 0,
        2, 0, 0, 3,
        0, 3, 2, 0,
    ], dtype=np.int64)
    v_partial = count_sudoku_violations(partial_4x4, n_size=4, box_r=2, box_c=2)
    assert v_partial == 0, f"Các ô [MASK] bị tính nhầm là vi phạm! {v_partial}"
    assert not is_valid_sudoku(partial_4x4, n_size=4, box_r=2, box_c=2), "Bảng còn [MASK] không được xem là hoàn tất!"
    print("  ✅ [PASS] Ô [MASK] (0) an toàn, không bị đếm sai.")

    # -----------------------------------------------------------------------
    # TEST 4: evaluate_predictions trên dự đoán 100% (Perfect Match)
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
    assert perfect_metrics["board_accuracy_pct"] == 100.0, "board_accuracy_pct phải là 100.0"
    assert perfect_metrics["cell_accuracy_pct"] == 100.0, "cell_accuracy_pct phải là 100.0"
    assert perfect_metrics["avg_constraint_violations"] == 0.0, "avg_constraint_violations phải là 0.0"
    assert perfect_metrics["valid_puzzle_rate"] == 1.0, "valid_puzzle_rate phải là 1.0"
    print("  ✅ [PASS] Perfect match: Board Acc = 100%, Cell Acc = 100%, Violations = 0.")

    # -----------------------------------------------------------------------
    # TEST 5: Cơ chế bảo vệ chia cho 0 (len > 0 else 0.0)
    # -----------------------------------------------------------------------
    print("\n[TEST 5] Kiểm tra cơ chế bảo vệ chia cho 0 khi không có ô nào thuộc nhóm margin...")
    empty_history_metrics = evaluate_predictions(
        preds_2d=perfect_preds,
        targets_2d=perfect_targets,
        histories=[],
        n_size=4,
        box_r=2,
        box_c=2,
    )
    assert empty_history_metrics["high_margin_cell_acc_pct"] == 0.0, "high_margin_cell_acc_pct khi rỗng phải là 0.0"
    assert empty_history_metrics["low_margin_cell_acc_pct"] == 0.0, "low_margin_cell_acc_pct khi rỗng phải là 0.0"

    empty_analysis = analyze_margin_vs_accuracy(step_histories=[], targets_2d=perfect_targets)
    assert empty_analysis["high_margin_cell_acc_pct"] == 0.0
    assert empty_analysis["low_margin_cell_acc_pct"] == 0.0
    print("  ✅ [PASS] Cơ chế bảo vệ chia cho 0 (len > 0 else 0.0) hoạt động an toàn tuyệt đối.")

    # -----------------------------------------------------------------------
    # TEST 6: analyze_margin_vs_accuracy và evaluate_predictions với Mock Data
    # -----------------------------------------------------------------------
    print("\n[TEST 6] Đánh giá chi tiết high_margin vs low_margin trên dữ liệu Mock 9x9...")
    quizzes, solutions = generate_mock_data(num_samples=10, n_size=9)
    mock_histories = [generate_mock_step_history(quizzes[i], n_size=9) for i in range(10)]
    mock_preds = np.array([h[-1]["board_state"] for h in mock_histories])

    metrics_res = evaluate_predictions(
        preds_2d=mock_preds,
        targets_2d=solutions,
        histories=mock_histories,
        n_size=9,
        box_r=3,
        box_c=3,
        threshold_high=0.5,
        threshold_low=0.2,
    )

    REQUIRED_KEYS = {
        "board_accuracy_pct",
        "cell_accuracy_pct",
        "avg_constraint_violations",
        "high_margin_cell_acc_pct",
        "low_margin_cell_acc_pct",
        "puzzle_accuracy",
        "cell_accuracy",
        "mean_violations",
        "total_violations",
        "valid_puzzle_rate",
        "mean_steps",
        "mean_margin",
        "num_samples",
    }
    assert REQUIRED_KEYS.issubset(set(metrics_res.keys())), "Thiếu các khóa chuẩn hóa trong evaluate_predictions!"
    assert 0.0 <= metrics_res["high_margin_cell_acc_pct"] <= 100.0
    assert 0.0 <= metrics_res["low_margin_cell_acc_pct"] <= 100.0

    margin_analysis = analyze_margin_vs_accuracy(
        step_histories=mock_histories,
        targets_2d=solutions,
        threshold_high=0.5,
        threshold_low=0.2,
    )
    assert "bins" in margin_analysis and len(margin_analysis["bins"]) == 5
    assert len(margin_analysis["bin_accuracies_pct"]) == 5
    assert margin_analysis["total_cells"] > 0
    print(f"  ✅ [PASS] Phân tích margin thành công trên {margin_analysis['total_cells']} ô unmasked:")
    print(f"      • High margin acc: {margin_analysis['high_margin_cell_acc_pct']:.2f}% (tổng {margin_analysis['high_margin_total']} ô)")
    print(f"      • Low margin acc : {margin_analysis['low_margin_cell_acc_pct']:.2f}% (tổng {margin_analysis['low_margin_total']} ô)")

    # -----------------------------------------------------------------------
    # TEST 7: aggregate_multi_seed_results qua 3 random seeds
    # -----------------------------------------------------------------------
    print("\n[TEST 7] Kiểm tra aggregate_multi_seed_results qua 3 random seeds...")
    seed_runs = [
        {"board_accuracy_pct": 80.0, "cell_accuracy_pct": 92.0, "avg_constraint_violations": 0.5},
        {"board_accuracy_pct": 85.0, "cell_accuracy_pct": 94.0, "avg_constraint_violations": 0.4},
        {"board_accuracy_pct": 82.0, "cell_accuracy_pct": 93.0, "avg_constraint_violations": 0.6},
    ]
    agg = aggregate_multi_seed_results(seed_runs)

    assert "board_accuracy_pct" in agg
    assert "board_accuracy_pct_mean" in agg
    assert "board_accuracy_pct_std" in agg
    assert agg["board_accuracy_pct"]["mean"] == agg["board_accuracy_pct_mean"]
    assert float(agg["board_accuracy_pct"]) == agg["board_accuracy_pct_mean"]
    # Mean của [80, 85, 82] = 82.3333...
    np.testing.assert_almost_equal(agg["board_accuracy_pct_mean"], 82.333333, decimal=4)
    print(f"  ✅ [PASS] Tổng hợp đa seed: board_accuracy_pct = {agg['board_accuracy_pct']}")
    print(f"      • Truy cập số float        : {float(agg['board_accuracy_pct']):.2f}")
    print(f"      • Truy cập ['mean']        : {agg['board_accuracy_pct']['mean']:.2f}")
    print(f"      • Truy cập ['std']         : {agg['board_accuracy_pct']['std']:.4f}")
    print(f"      • Truy cập flat '_mean'    : {agg['board_accuracy_pct_mean']:.2f}")

    # -----------------------------------------------------------------------
    # TEST 8: Vẽ biểu đồ và xuất file PNG/PDF
    # -----------------------------------------------------------------------
    print("\n[TEST 8] Kiểm tra xuất biểu đồ PNG và PDF...")
    import tempfile
    output_dir = os.path.join(tempfile.gettempdir(), "tv4_reports_figures")
    os.makedirs(output_dir, exist_ok=True)

    strategy_comparison_data = {
        "Random": {"board_accuracy_pct": 25.0, "cell_accuracy_pct": 74.0, "avg_constraint_violations": 3.8},
        "Top-Prob": {"board_accuracy_pct": 60.0, "cell_accuracy_pct": 89.0, "avg_constraint_violations": 1.4},
        "Margin (Ours)": {"board_accuracy_pct": 85.0, "cell_accuracy_pct": 96.5, "avg_constraint_violations": 0.3},
        "CSP Baseline": {"board_accuracy_pct": 100.0, "cell_accuracy_pct": 100.0, "avg_constraint_violations": 0.0},
    }

    png_strat = os.path.join(output_dir, "test_strategy_comparison.png")
    pdf_strat = os.path.join(output_dir, "test_strategy_comparison.pdf")
    fig_strat = plot_strategy_comparison(strategy_comparison_data, save_path=png_strat)
    plot_strategy_comparison(strategy_comparison_data, save_path=pdf_strat)

    png_margin = os.path.join(output_dir, "test_margin_vs_accuracy.png")
    pdf_margin = os.path.join(output_dir, "test_margin_vs_accuracy.pdf")
    fig_margin = plot_margin_vs_accuracy(margin_analysis, save_path=png_margin)
    plot_margin_vs_accuracy(margin_analysis, save_path=pdf_margin)

    assert os.path.exists(png_strat), f"Không tìm thấy file {png_strat}"
    assert os.path.exists(pdf_strat), f"Không tìm thấy file {pdf_strat}"
    assert os.path.exists(png_margin), f"Không tìm thấy file {png_margin}"
    assert os.path.exists(pdf_margin), f"Không tìm thấy file {pdf_margin}"
    print(f"  ✅ [PASS] Đã tạo thành công các tệp PNG và PDF tại: {output_dir}")

    # In báo cáo mẫu
    print("\n")
    print_evaluation_report(metrics_res, title="Báo Cáo Nghiệm Thu Toàn Diện TV4 (Ngày 1 & 2)")

    print("\n" + "=" * 72)
    print("🎉 TẤT CẢ 8 BỘ KIỂM THỬ CỦA TV4 ĐÃ VƯỢT QUA 100%! MODULE HOÀN THIỆN ĐẦY ĐỦ.")
    print("=" * 72)
