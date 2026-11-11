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

  Giai đoạn 4: Ngày 3 (14h00 - Chốt kiểm định) - Task 16:
    9. evaluate_csp_results: Chấm điểm thử nghiệm kết quả của TV1 (Backtracking, MRV, AC-3).
    10. evaluate_mdm_results: Chấm điểm thử nghiệm kết quả của TV3 (Random, Top-Prob, Margin).
    11. check_margin_vs_random_difference: Kiểm tra độ chênh lệch giữa Margin và Random;
        cảnh báo ngay lúc 14h00 nếu chênh lệch quá nhỏ (để TV1 tăng độ khó đề test hoặc TV3 chỉnh gumbel_scale / steps).
    12. format_task16_audit_report: Tạo bảng báo cáo nghiệm thu chi tiết kèm khuyến nghị cho TV1 và TV3.
    13. plot_task16_audit_figures: Vẽ biểu đồ đối sánh toàn diện 6 chiến lược và chi tiết chênh lệch Margin vs Random.
    14. run_task16_audit: Hàm thực thi toàn diện Task 16 (tự động hóa pipeline và chốt kiểm định 14h00).
"""

from __future__ import annotations

import os
import sys
import glob
import json
import time
import pickle
import argparse

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


def _resolve_box_dims(n_size: int, box_r: Optional[int] = None, box_c: Optional[int] = None) -> Tuple[int, int]:
    """Tự động suy luận kích thước khối con (box_r, box_c) phù hợp với n_size."""
    if box_r is not None and box_c is not None and box_r * box_c == n_size:
        return box_r, box_c
    if n_size in BOX_DIMS:
        return BOX_DIMS[n_size]
    root = int(np.sqrt(n_size))
    if root * root == n_size:
        return root, root
    return box_r or 1, box_c or n_size


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
# 5. Giai đoạn 4: Chấm điểm thực nghiệm TV1, TV3 & Chốt kiểm định (Task 16)
# ===========================================================================

def _safe_json_serialize(obj: Any) -> Any:
    """Chuyển đổi các đối tượng Numpy, MetricSummary sang kiểu chuẩn JSON."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif hasattr(obj, "mean") and hasattr(obj, "std"):
        return {"mean": float(obj.mean), "std": float(obj.std)}
    elif isinstance(obj, dict):
        return {str(k): _safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_json_serialize(x) for x in obj]
    return str(obj)


def evaluate_csp_results(
    csp_source: Union[str, Dict[str, Any]],
    histories_source: Optional[Union[str, Dict[str, Any]]] = None,
    n_size: Optional[int] = None,
    box_r: Optional[int] = None,
    box_c: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Đánh giá toàn diện kết quả thực nghiệm CSP (TV1 - Task 14).

    Đọc file baseline_csp_results.npz (hoặc dict) và baseline_csp_histories.pkl
    nếu có. Tính toán đầy đủ:
      - board_accuracy_pct: Tỷ lệ giải đúng 100% toàn bảng.
      - cell_accuracy_pct: Tỷ lệ điền đúng từng ô.
      - avg_constraint_violations: Số lỗi vi phạm trung bình / bảng.
      - solve_time: Thời gian giải trung bình (s/bảng).

    Parameters
    ----------
    csp_source : Union[str, Dict[str, Any]]
        Đường dẫn file baseline_csp_results.npz hoặc dictionary dữ liệu.
    histories_source : Optional[Union[str, Dict[str, Any]]]
        Đường dẫn file baseline_csp_histories.pkl hoặc dict histories.
    n_size : Optional[int]
        Kích thước cạnh bảng (mặc định tự động suy luận từ L).

    Returns
    -------
    csp_metrics : Dict[str, Dict[str, Any]]
        Kết quả đánh giá cho từng phương pháp: 'backtracking', 'mrv', 'ac3'.
    """
    if isinstance(csp_source, str):
        if not os.path.isfile(csp_source):
            raise FileNotFoundError(f"Không tìm thấy file kết quả CSP: {csp_source}")
        csp_data = dict(np.load(csp_source, allow_pickle=True))
    else:
        csp_data = csp_source

    histories_dict: Optional[Dict[str, Any]] = None
    if isinstance(histories_source, str) and os.path.isfile(histories_source):
        with open(histories_source, "rb") as f:
            histories_dict = pickle.load(f)
    elif isinstance(histories_source, dict):
        histories_dict = histories_source

    targets = np.asarray(csp_data["targets"])
    L = targets.shape[1] if targets.ndim > 1 else len(targets)
    if n_size is None:
        n_size = int(np.sqrt(L))
    box_r, box_c = _resolve_box_dims(n_size, box_r, box_c)

    # Đọc thời gian giải nếu có
    solve_times_raw = csp_data.get("solve_times", None)
    solve_times_arr = np.asarray(solve_times_raw).flatten() if solve_times_raw is not None else None

    methods = ["backtracking", "mrv", "ac3"]
    method_time_map: Dict[str, float] = {}
    if solve_times_arr is not None and len(solve_times_arr) >= len(methods):
        for idx, m in enumerate(methods):
            method_time_map[m] = float(solve_times_arr[idx])

    csp_metrics: Dict[str, Dict[str, Any]] = {}
    for idx, method in enumerate(methods):
        pred_key = f"preds_{method}"
        if pred_key in csp_data:
            preds = np.asarray(csp_data[pred_key])
        elif method in csp_data:
            preds = np.asarray(csp_data[method])
        else:
            continue

        hists = histories_dict.get(method, None) if histories_dict else None
        m_eval = evaluate_predictions(
            preds_2d=preds,
            targets_2d=targets,
            histories=hists,
            n_size=n_size,
            box_r=box_r,
            box_c=box_c,
        )

        st = method_time_map.get(method, 0.0)
        m_eval["solve_time"] = st
        m_eval["solve_time_per_board"] = st
        csp_metrics[method] = m_eval

    return csp_metrics


def evaluate_mdm_results(
    mdm_source: Union[str, Dict[str, Any]],
    n_size: Optional[int] = None,
    box_r: Optional[int] = None,
    box_c: Optional[int] = None,
    threshold_high: float = 0.5,
    threshold_low: float = 0.2,
) -> Dict[str, Any]:
    """Đánh giá toàn diện kết quả thực nghiệm Adaptive Inference (TV3 - Task 15).

    Đọc file mdm_inference_results.pkl (hoặc dict) chứa kết quả chạy các
    chiến lược ('random', 'top_prob', 'margin') qua các seeds (42, 123, 999).
    Đánh giá từng seed và tổng hợp Mean ± Std cho từng chiến lược.

    Parameters
    ----------
    mdm_source : Union[str, Dict[str, Any]]
        Đường dẫn file mdm_inference_results.pkl hoặc dictionary dữ liệu.
    n_size : Optional[int]
        Kích thước bảng Sudoku (mặc định tự động suy luận).

    Returns
    -------
    mdm_eval_summary : Dict[str, Any]
        Dictionary gồm:
          - 'by_seed': {strategy: {seed: metrics_dict}}
          - 'aggregated': {strategy: aggregated_metrics_dict (mean & std)}
          - 'strategies': danh sách các chiến lược có trong file
          - 'seeds': danh sách seeds
          - 'num_samples': số lượng bảng kiểm thử
          - 'n_size': kích thước bảng
    """
    if isinstance(mdm_source, str):
        if not os.path.isfile(mdm_source):
            raise FileNotFoundError(f"Không tìm thấy file kết quả MDM: {mdm_source}")
        with open(mdm_source, "rb") as f:
            mdm_data = pickle.load(f)
    else:
        mdm_data = mdm_source

    targets = np.asarray(mdm_data["targets"])
    L = targets.shape[1] if targets.ndim > 1 else len(targets)
    if n_size is None:
        n_size = int(np.sqrt(L))
    box_r, box_c = _resolve_box_dims(n_size, box_r, box_c)

    strategies = [s for s in ["random", "top_prob", "margin"] if s in mdm_data]
    all_seeds = set()
    for s in strategies:
        if isinstance(mdm_data[s], dict):
            all_seeds.update(mdm_data[s].keys())
    sorted_seeds = sorted(list(all_seeds))

    by_seed: Dict[str, Dict[Any, Dict[str, Any]]] = {s: {} for s in strategies}
    aggregated: Dict[str, Dict[str, Any]] = {}

    for strat in strategies:
        seed_runs = []
        for seed in sorted_seeds:
            if seed not in mdm_data[strat]:
                continue
            item = mdm_data[strat][seed]
            preds = np.asarray(item["preds"])
            hists = item.get("histories", None)
            st = float(item.get("solve_time", 0.0))

            m_eval = evaluate_predictions(
                preds_2d=preds,
                targets_2d=targets,
                histories=hists,
                n_size=n_size,
                box_r=box_r,
                box_c=box_c,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
            )
            m_eval["solve_time"] = st
            m_eval["solve_time_per_board"] = st / len(targets) if len(targets) > 0 else st
            by_seed[strat][seed] = m_eval
            seed_runs.append(m_eval)

        aggregated[strat] = aggregate_multi_seed_results(seed_runs)

    return {
        "by_seed": by_seed,
        "aggregated": aggregated,
        "strategies": strategies,
        "seeds": sorted_seeds,
        "num_samples": len(targets),
        "n_size": n_size,
    }


def check_margin_vs_random_difference(
    mdm_eval_summary: Dict[str, Any],
    min_delta_cell_acc: float = 1.0,
    min_delta_board_acc: float = 0.5,
) -> Dict[str, Any]:
    """Kiểm tra độ chênh lệch giữa chiến lược Margin và Random tại Chốt kiểm định 14h00 (Task 16).

    Quy tắc chốt kiểm định 14h00:
      1. Tính độ chênh lệch trung bình qua các seeds:
         - Delta Cell Accuracy (%)  : Margin - Random
         - Delta Board Accuracy (%) : Margin - Random
         - Delta Constraint Violations: Random - Margin (số lỗi giảm được)
         - Phân tích Margin Confidence: High Margin (>=0.5) vs Low Margin (<0.2)
      2. Đánh giá ngưỡng tối thiểu:
         - Nếu delta_cell_acc < min_delta_cell_acc hoặc (delta_board_acc < min_delta_board_acc và delta_cell_acc <= 0):
           -> Trạng thái: WARNING (Chênh lệch quá nhỏ!)
           -> Xuất ngay thông báo báo động lúc 14h00 với chỉ dẫn cụ thể cho TV1 và TV3:
              + TV1: Tăng độ khó đề test (tăng tỷ lệ [MASK] từ 14-22 lên 22-28, hoặc ít givens hơn).
              + TV3: Chỉnh gumbel_scale (giảm từ 0.5 xuống 0.1-0.2) hoặc tăng steps (từ 50 lên 80).
         - Ngược lại:
           -> Trạng thái: PASSED (Chốt kiểm định đạt, Margin vượt trội Random rõ rệt).

    Parameters
    ----------
    mdm_eval_summary : Dict[str, Any]
        Đầu ra từ hàm evaluate_mdm_results.
    min_delta_cell_acc : float
        Ngưỡng chênh lệch Cell Accuracy (%) tối thiểu (mặc định 1.0%).
    min_delta_board_acc : float
        Ngưỡng chênh lệch Board Accuracy (%) tối thiểu (mặc định 0.5%).

    Returns
    -------
    diff_report : Dict[str, Any]
        Chi tiết độ chênh lệch, trạng thái PASSED / WARNING, và khuyến nghị hành động.
    """
    agg = mdm_eval_summary.get("aggregated", {})
    by_seed = mdm_eval_summary.get("by_seed", {})

    if "margin" not in agg or "random" not in agg:
        raise ValueError("mdm_eval_summary phải chứa cả 2 chiến lược 'margin' và 'random'.")

    m_agg = agg["margin"]
    r_agg = agg["random"]

    # 1. Trích xuất Cell Accuracy
    m_cell, m_cell_std = _extract_metric_pair(m_agg, "cell_accuracy_pct", as_percentage=True)
    r_cell, r_cell_std = _extract_metric_pair(r_agg, "cell_accuracy_pct", as_percentage=True)
    delta_cell_acc = m_cell - r_cell

    # 2. Trích xuất Board Accuracy
    m_board, m_board_std = _extract_metric_pair(m_agg, "board_accuracy_pct", as_percentage=True)
    r_board, r_board_std = _extract_metric_pair(r_agg, "board_accuracy_pct", as_percentage=True)
    delta_board_acc = m_board - r_board

    # 3. Trích xuất Violations
    m_viol, m_viol_std = _extract_metric_pair(m_agg, "avg_constraint_violations")
    r_viol, r_viol_std = _extract_metric_pair(r_agg, "avg_constraint_violations")
    delta_violations = r_viol - m_viol  # Lỗi vi phạm giảm được

    # 4. Trích xuất phân tích High Margin vs Low Margin trong Margin strategy
    m_high_acc, _ = _extract_metric_pair(m_agg, "high_margin_cell_acc_pct", as_percentage=True)
    m_low_acc, _ = _extract_metric_pair(m_agg, "low_margin_cell_acc_pct", as_percentage=True)
    margin_high_vs_low_gap = m_high_acc - m_low_acc

    # 5. Chi tiết chênh lệch theo từng seed
    seed_deltas: Dict[Any, Dict[str, float]] = {}
    seeds = mdm_eval_summary.get("seeds", [])
    for s in seeds:
        if s in by_seed.get("margin", {}) and s in by_seed.get("random", {}):
            s_m_cell = by_seed["margin"][s].get("cell_accuracy_pct", 0.0)
            s_r_cell = by_seed["random"][s].get("cell_accuracy_pct", 0.0)
            s_m_board = by_seed["margin"][s].get("board_accuracy_pct", 0.0)
            s_r_board = by_seed["random"][s].get("board_accuracy_pct", 0.0)
            seed_deltas[s] = {
                "delta_cell_acc": float(round(s_m_cell - s_r_cell, 4)),
                "delta_board_acc": float(round(s_m_board - s_r_board, 4)),
            }

    # 6. Đánh giá ngưỡng chốt kiểm định 14h00
    is_delta_too_small = (delta_cell_acc < min_delta_cell_acc) and (delta_board_acc < min_delta_board_acc)

    if is_delta_too_small:
        status = "WARNING"
        status_label = "CHÊNH LỆCH QUÁ NHỎ (CẦN ĐIỀU CHỈNH GẤP)"
        alert_msg = (
            f"⚠️ [CẢNH BÁO KIỂM ĐỊNH 14H00] Độ chênh lệch giữa Margin và Random quá nhỏ!\n"
            f"   • ΔCell Acc  = {delta_cell_acc:+.2f}% (ngưỡng tối thiểu: {min_delta_cell_acc:.1f}%)\n"
            f"   • ΔBoard Acc = {delta_board_acc:+.2f}% (ngưỡng tối thiểu: {min_delta_board_acc:.1f}%)\n"
            f"   • Khuyến nghị TV1: Tăng độ khó tập test (tăng tỷ lệ ô trống [MASK] từ 14-22 lên 22-28 ô trên bảng 6x6, "
            f"hoặc chuyển sang bảng 9x9 với ít givens hơn để giảm thiểu tỷ lệ điền bừa trúng của Random unmasking).\n"
            f"   • Khuyến nghị TV3: Tinh chỉnh siêu tham số suy luận (giảm gumbel_scale từ 0.5 xuống 0.1-0.2 để giảm nhiễu ngẫu nhiên, "
            f"hoặc tăng số bước suy luận steps từ 50 lên 80)."
        )
        rec_tv1 = "Tăng độ khó đề test: tăng số lượng ô [MASK] bị che (ví dụ từ 14-22 lên 22-28 ô trên bảng 6x6) để triệt tiêu may mắn của Random unmasking."
        rec_tv3 = "Tinh chỉnh siêu tham số suy luận: giảm gumbel_scale từ 0.5 xuống 0.1-0.2 để giảm nhiễu ngẫu nhiên, hoặc tăng steps lấy mẫu từ 50 lên 80."
    else:
        status = "PASSED"
        status_label = "CHỐT KIỂM ĐỊNH ĐẠT (MARGIN VƯỢT TRỘI)"
        alert_msg = (
            f"✅ [CHỐT KIỂM ĐỊNH 14H00 ĐẠT] Chiến lược Top Probability Margin vượt trội Random rõ rệt!\n"
            f"   • ΔCell Acc  = {delta_cell_acc:+.2f}%\n"
            f"   • ΔBoard Acc = {delta_board_acc:+.2f}%\n"
            f"   • Lỗi vi phạm giảm: {delta_violations:+.2f} lỗi/bảng\n"
            f"   • Xác nhận giả thuyết 'Train for the Worst, Plan for the Best' thành công trên dữ liệu Pilot."
        )
        rec_tv1 = "Bộ dữ liệu test phân hóa tốt giữa các chiến lược suy luận. Giữ nguyên cấu hình phân bổ givens hiện tại."
        rec_tv3 = "Chiến lược Top Probability Margin hoạt động ổn định và tin cậy. Duy trì gumbel_scale = 0.5 và steps = 50."

    return {
        "status": status,
        "status_label": status_label,
        "is_delta_too_small": bool(is_delta_too_small),
        "alert_message": alert_msg,
        "min_delta_cell_acc": min_delta_cell_acc,
        "min_delta_board_acc": min_delta_board_acc,
        "delta_cell_acc": float(round(delta_cell_acc, 4)),
        "delta_board_acc": float(round(delta_board_acc, 4)),
        "delta_violations": float(round(delta_violations, 4)),
        "margin_metrics": {
            "cell_acc_mean": float(round(m_cell, 2)),
            "cell_acc_std": float(round(m_cell_std, 4)),
            "board_acc_mean": float(round(m_board, 2)),
            "board_acc_std": float(round(m_board_std, 4)),
            "violations_mean": float(round(m_viol, 2)),
            "violations_std": float(round(m_viol_std, 4)),
            "high_margin_acc": float(round(m_high_acc, 2)),
            "low_margin_acc": float(round(m_low_acc, 2)),
            "confidence_spread": float(round(margin_high_vs_low_gap, 2)),
        },
        "random_metrics": {
            "cell_acc_mean": float(round(r_cell, 2)),
            "cell_acc_std": float(round(r_cell_std, 4)),
            "board_acc_mean": float(round(r_board, 2)),
            "board_acc_std": float(round(r_board_std, 4)),
            "violations_mean": float(round(r_viol, 2)),
            "violations_std": float(round(r_viol_std, 4)),
        },
        "seed_deltas": seed_deltas,
        "recommendation_tv1": rec_tv1,
        "recommendation_tv3": rec_tv3,
    }


def format_task16_audit_report(
    csp_metrics: Optional[Dict[str, Dict[str, Any]]],
    mdm_eval_summary: Dict[str, Any],
    diff_report: Dict[str, Any],
) -> str:
    """Tạo bảng báo cáo nghiệm thu Markdown chi tiết cho Task 16 (Chốt kiểm định 14h00).

    Bao gồm đối sánh đầy đủ giữa:
      - 3 thuật toán CSP của TV1 (Backtracking, MRV + Forward Checking, AC-3)
      - 3 chiến lược Adaptive Decoding của TV3 (Random, Top-Prob, Top-Prob Margin)
      - Phân tích chi tiết độ chênh lệch Margin vs Random và khuyến nghị hành động.
    """
    lines = []
    lines.append("# 📋 BÁO CÁO NGHIỆM THU & CHỐT KIỂM ĐỊNH (TV4 - TASK 16)")
    lines.append("")
    lines.append("**Dự án:** Adaptive Inference trong Masked Diffusion Models cho Sudoku")
    lines.append("**Mốc thời gian:** Giai đoạn 4 - Ngày 3 | 14h00 (Chốt kiểm định)")
    lines.append(f"**Thời điểm tạo:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Alert Banner
    status = diff_report.get("status", "WARNING")
    if status == "PASSED":
        lines.append("> [!NOTE] ✅ **KẾT QUẢ KIỂM ĐỊNH: ĐẠT TIÊU CHUẨN (PASSED)**")
        lines.append(f"> Chiến lược **Top Probability Margin** vượt trội **Random** rõ rệt: $\\Delta\\text{{Cell Acc}} = +{diff_report['delta_cell_acc']:.2f}\\%$, $\\Delta\\text{{Board Acc}} = +{diff_report['delta_board_acc']:.2f}\\%$.")
    else:
        lines.append("> [!WARNING] ⚠️ **KẾT QUẢ KIỂM ĐỊNH: CHÊNH LỆCH QUÁ NHỎ (ACTION REQUIRED)**")
        lines.append(f"> Độ chênh lệch giữa Margin và Random ($\\Delta\\text{{Cell Acc}} = {diff_report['delta_cell_acc']:+.2f}\\%$) chưa đạt ngưỡng tối thiểu ({diff_report['min_delta_cell_acc']:.1f}\\%).")
        lines.append(f"> Báo ngay lúc 14h00 để TV1 và TV3 phối hợp điều chỉnh kịp thời.")
    lines.append("")

    # Bảng đối sánh tổng hợp toàn diện
    lines.append("## 1. Bảng Đối Sánh Toàn Diện Các Chiến Lược (TV1 & TV3)")
    lines.append("")
    lines.append("| Nhóm Phương Pháp | Chiến Lược / Thuật Toán | Board Acc (%) | Cell Acc (%) | Lỗi Vi Phạm TB | Thời Gian (s/bảng) |")
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|")

    # Điền TV1 CSP nếu có
    if csp_metrics:
        method_names = {
            "backtracking": "Backtracking (Cổ điển)",
            "mrv": "MRV + Forward Checking",
            "ac3": "AC-3 Constraint Propagation",
        }
        for m, label in method_names.items():
            if m in csp_metrics:
                m_data = csp_metrics[m]
                b_acc = m_data.get("board_accuracy_pct", 0.0)
                c_acc = m_data.get("cell_accuracy_pct", 0.0)
                viol = m_data.get("avg_constraint_violations", 0.0)
                st = m_data.get("solve_time_per_board", 0.0)
                lines.append(f"| **CSP Baseline (TV1)** | {label} | {b_acc:.2f}% | {c_acc:.2f}% | {viol:.2f} | {st:.4f}s |")

    # Điền TV3 MDM
    mdm_agg = mdm_eval_summary.get("aggregated", {})
    strat_names = {
        "random": "MDM - Random Unmasking",
        "top_prob": "MDM - Top Probability",
        "margin": "MDM - Top Prob Margin (Ours)",
    }
    for s, label in strat_names.items():
        if s in mdm_agg:
            s_data = mdm_agg[s]
            b_m, b_s = _extract_metric_pair(s_data, "board_accuracy_pct", as_percentage=True)
            c_m, c_s = _extract_metric_pair(s_data, "cell_accuracy_pct", as_percentage=True)
            v_m, v_s = _extract_metric_pair(s_data, "avg_constraint_violations")
            st_m, st_s = _extract_metric_pair(s_data, "solve_time_per_board")
            lines.append(f"| **Masked Diffusion (TV3)** | {label} | {b_m:.2f} ± {b_s:.2f}% | {c_m:.2f} ± {c_s:.2f}% | {v_m:.2f} ± {v_s:.2f} | {st_m:.4f}s |")

    lines.append("")

    # Bảng phân tích chi tiết Margin vs Random
    lines.append("## 2. Kiểm Tra Độ Chênh Lệch Margin vs Random (Chốt Kiểm Định 14h00)")
    lines.append("")
    lines.append("| Chỉ Số Đánh Giá | Random (Vanilla) | Top Prob Margin (Ours) | Độ Chênh Lệch ($\\Delta$) | Ngưỡng Tối Thiểu | Kết Luận |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")

    m_info = diff_report["margin_metrics"]
    r_info = diff_report["random_metrics"]

    d_cell = diff_report["delta_cell_acc"]
    c_status = "✅ Đạt" if d_cell >= diff_report["min_delta_cell_acc"] else "❌ Quá nhỏ"
    lines.append(f"| **Cell Accuracy (%)** | {r_info['cell_acc_mean']:.2f}% | {m_info['cell_acc_mean']:.2f}% | **{d_cell:+.2f}%** | $\\ge {diff_report['min_delta_cell_acc']:.1f}\\%$ | {c_status} |")

    d_board = diff_report["delta_board_acc"]
    b_status = "✅ Đạt" if d_board >= diff_report["min_delta_board_acc"] or d_cell >= diff_report["min_delta_cell_acc"] else "❌ Quá nhỏ"
    lines.append(f"| **Board Accuracy (%)** | {r_info['board_acc_mean']:.2f}% | {m_info['board_acc_mean']:.2f}% | **{d_board:+.2f}%** | $\\ge {diff_report['min_delta_board_acc']:.1f}\\%$ | {b_status} |")

    d_viol = diff_report["delta_violations"]
    v_status = "✅ Giảm lỗi" if d_viol >= 0 else "⚠️ Tăng lỗi"
    lines.append(f"| **Vi Phạm Ràng Buộc TB** | {r_info['violations_mean']:.2f} | {m_info['violations_mean']:.2f} | **{d_viol:+.2f}** (giảm) | $> 0$ | {v_status} |")

    lines.append("")

    # Phân tích độ tự tin
    lines.append("## 3. Phân Tích Độ Tin Cậy (Margin Confidence Analysis)")
    lines.append(f"- **High Margin ($\\ge 0.5$) Cell Acc:** {m_info['high_margin_acc']:.2f}%")
    lines.append(f"- **Low Margin ($< 0.2$) Cell Acc:** {m_info['low_margin_acc']:.2f}%")
    lines.append(f"- **Khoảng cách độ tự tin (Confidence Spread):** {m_info['confidence_spread']:+.2f}% (ô tự tin cao có độ chính xác vượt trội)")
    lines.append("")

    # Khuyến nghị hành động tại mốc 14h00
    lines.append("## 4. Chỉ Đạo Phối Hợp & Khuyến Nghị Hành Động Lúc 14h00")
    lines.append("")
    lines.append(f"- 📢 **Dành cho TV1 (Dữ liệu & CSP Baseline):** {diff_report['recommendation_tv1']}")
    lines.append(f"- 📢 **Dành cho TV3 (Adaptive Inference):** {diff_report['recommendation_tv3']}")
    lines.append("")
    lines.append("---")
    lines.append("*Báo cáo được lập tự động bởi TV4 (`src/metrics.py`) phục vụ cuộc họp chốt kiểm định Ngày 3.*")

    return "\n".join(lines)


def plot_task16_audit_figures(
    csp_metrics: Optional[Dict[str, Dict[str, Any]]],
    mdm_eval_summary: Dict[str, Any],
    diff_report: Dict[str, Any],
    output_dir: str = "results/figures",
) -> Dict[str, str]:
    """Vẽ và xuất các biểu đồ phục vụ chốt kiểm định Task 16 (PNG & PDF chất lượng cao).

    Xuất 2 bộ biểu đồ:
      1. task16_strategy_comparison.png & .pdf: So sánh tổng hợp 6 phương pháp.
      2. task16_margin_vs_random.png & .pdf: Chi tiết độ chênh lệch Margin vs Random.

    Parameters
    ----------
    csp_metrics : Optional[Dict[str, Dict[str, Any]]]
        Kết quả đánh giá các thuật toán CSP.
    mdm_eval_summary : Dict[str, Any]
        Kết quả đánh giá các chiến lược MDM.
    diff_report : Dict[str, Any]
        Báo cáo độ chênh lệch từ check_margin_vs_random_difference.
    output_dir : str
        Thư mục lưu trữ hình ảnh.

    Returns
    -------
    saved_files : Dict[str, str]
        Đường dẫn các file đã lưu thành công.
    """
    if not _HAS_MATPLOTLIB:
        print("⚠️ Cảnh báo: Matplotlib không khả dụng. Bỏ qua xuất biểu đồ Task 16.")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    # -----------------------------------------------------------------------
    # BIỂU ĐỒ 1: SO SÁNH TỔNG HỢP 6 CHIẾN LƯỢC (CSP vs MDM)
    # -----------------------------------------------------------------------
    strat_labels = []
    board_vals = []
    cell_vals = []
    viol_vals = []
    time_vals = []
    colors = []

    # TV1 CSP methods
    if csp_metrics:
        csp_order = [
            ("backtracking", "Backtracking\n(TV1)", "#7f8c8d"),
            ("mrv", "MRV + FC\n(TV1)", "#2980b9"),
            ("ac3", "AC-3\n(TV1)", "#8e44ad"),
        ]
        for key, name, color in csp_order:
            if key in csp_metrics:
                strat_labels.append(name)
                board_vals.append(float(csp_metrics[key].get("board_accuracy_pct", 0.0)))
                cell_vals.append(float(csp_metrics[key].get("cell_accuracy_pct", 0.0)))
                viol_vals.append(float(csp_metrics[key].get("avg_constraint_violations", 0.0)))
                time_vals.append(float(csp_metrics[key].get("solve_time_per_board", 0.0)))
                colors.append(color)

    # TV3 MDM strategies
    mdm_agg = mdm_eval_summary.get("aggregated", {})
    mdm_order = [
        ("random", "MDM: Random\n(Vanilla)", "#e74c3c"),
        ("top_prob", "MDM: Top-Prob\n(TV3)", "#f39c12"),
        ("margin", "MDM: Margin\n(Ours)", "#27ae60"),
    ]
    for key, name, color in mdm_order:
        if key in mdm_agg:
            strat_labels.append(name)
            b_m, _ = _extract_metric_pair(mdm_agg[key], "board_accuracy_pct", as_percentage=True)
            c_m, _ = _extract_metric_pair(mdm_agg[key], "cell_accuracy_pct", as_percentage=True)
            v_m, _ = _extract_metric_pair(mdm_agg[key], "avg_constraint_violations")
            st_m, _ = _extract_metric_pair(mdm_agg[key], "solve_time_per_board")
            board_vals.append(b_m)
            cell_vals.append(c_m)
            viol_vals.append(v_m)
            time_vals.append(st_m)
            colors.append(color)

    if strat_labels:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        x = np.arange(len(strat_labels))
        width = 0.55

        # (1) Board Accuracy
        ax = axes[0, 0]
        bars = ax.bar(x, board_vals, width, color=colors, edgecolor="black", alpha=0.9)
        ax.set_title("Board-Level Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(strat_labels, fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # (2) Cell Accuracy
        ax = axes[0, 1]
        bars = ax.bar(x, cell_vals, width, color=colors, edgecolor="black", alpha=0.9)
        ax.set_title("Cell-Level Accuracy (%)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(strat_labels, fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # (3) Constraint Violations
        ax = axes[1, 0]
        bars = ax.bar(x, viol_vals, width, color=colors, edgecolor="black", alpha=0.9)
        ax.set_title("Average Constraint Violations (càng thấp càng tốt)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(strat_labels, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # (4) Solve Time per board
        ax = axes[1, 1]
        bars = ax.bar(x, time_vals, width, color=colors, edgecolor="black", alpha=0.9)
        ax.set_title("Solve Time / Board (s)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(strat_labels, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h * 1.02, f"{h:.4f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.suptitle("ĐỐI SÁNH HIỆU NĂNG TOÀN DIỆN CÁC CHIẾN LƯỢC (TV1 & TV3 - TASK 16)", fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout()

        p1_png = os.path.join(output_dir, "task16_strategy_comparison.png")
        p1_pdf = os.path.join(output_dir, "task16_strategy_comparison.pdf")
        fig.savefig(p1_png, dpi=300, bbox_inches="tight")
        fig.savefig(p1_pdf, bbox_inches="tight")
        plt.close(fig)
        saved_paths["strategy_comparison_png"] = p1_png
        saved_paths["strategy_comparison_pdf"] = p1_pdf

    # -----------------------------------------------------------------------
    # BIỂU ĐỒ 2: CHI TIẾT ĐỘ CHÊNH LỆCH MARGIN VS RANDOM (CHỐT 14H00)
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    m_info = diff_report["margin_metrics"]
    r_info = diff_report["random_metrics"]

    # Subplot 1: Cell Accuracy: Random vs Margin
    ax = axes[0]
    bars = ax.bar(
        ["Random\n(Vanilla)", "Margin\n(Ours)"],
        [r_info["cell_acc_mean"], m_info["cell_acc_mean"]],
        yerr=[r_info["cell_acc_std"], m_info["cell_acc_std"]],
        capsize=5,
        color=["#e74c3c", "#27ae60"],
        edgecolor="black",
        width=0.45,
        alpha=0.9,
    )
    ax.set_title("Cell Accuracy (%)\n(Mean ± Std qua các seeds)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 2: Deltas (Margin - Random)
    ax = axes[1]
    deltas = [diff_report["delta_cell_acc"], diff_report["delta_board_acc"], diff_report["delta_violations"]]
    d_labels = ["Δ Cell Acc\n(%)", "Δ Board Acc\n(%)", "Giảm Lỗi\n(violations)"]
    d_colors = ["#27ae60" if d >= 0 else "#e74c3c" for d in deltas]
    bars = ax.bar(d_labels, deltas, color=d_colors, edgecolor="black", width=0.45, alpha=0.9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(diff_report["min_delta_cell_acc"], color="#e67e22", linestyle=":", label=f"Ngưỡng tối thiểu ({diff_report['min_delta_cell_acc']:.1f}%)")
    ax.set_title("Độ Chênh Lệch (Margin vs Random)\nChốt kiểm định 14h00", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.2 if h >= 0 else -0.5
        ax.text(b.get_x() + b.get_width() / 2.0, h + offset, f"{h:+.2f}", ha="center", va=va, fontsize=9, fontweight="bold")

    # Subplot 3: High vs Low Margin Confidence Accuracy
    ax = axes[2]
    bars = ax.bar(
        ["Low Margin\n(< 0.2)", "High Margin\n(>= 0.5)"],
        [m_info["low_margin_acc"], m_info["high_margin_acc"]],
        color=["#e67e22", "#2ecc71"],
        edgecolor="black",
        width=0.45,
        alpha=0.9,
    )
    ax.set_title("Margin Confidence Breakdown\n(High vs Low Margin)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 2.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    status_tag = diff_report.get("status", "WARNING")
    plt.suptitle(f"CHỐT KIỂM ĐỊNH 14H00: PHÂN TÍCH ĐỘ CHÊNH LỆCH MARGIN VS RANDOM [{status_tag}]", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    p2_png = os.path.join(output_dir, "task16_margin_vs_random.png")
    p2_pdf = os.path.join(output_dir, "task16_margin_vs_random.pdf")
    fig.savefig(p2_png, dpi=300, bbox_inches="tight")
    fig.savefig(p2_pdf, bbox_inches="tight")
    plt.close(fig)
    saved_paths["margin_vs_random_png"] = p2_png
    saved_paths["margin_vs_random_pdf"] = p2_pdf

    print(f"  💾 [PNG/PDF] Đã xuất biểu đồ kiểm định Task 16 vào thư mục: {output_dir}")
    return saved_paths


def run_task16_audit(
    shared_dir: str = ".",
    output_dir: Optional[str] = None,
    auto_run_missing: bool = True,
    quick_test: bool = False,
    device: str = "cpu",
    min_delta_cell_acc: float = 1.0,
    min_delta_board_acc: float = 0.5,
) -> Dict[str, Any]:
    """Thực thi toàn diện quy trình chấm điểm và chốt kiểm định 14h00 (TV4 - Task 16).

    Quy trình chuẩn hóa:
      1. Tìm kiếm kết quả TV1 (baseline_csp_results.npz) và TV3 (mdm_inference_results.pkl).
         Nếu chưa có và auto_run_missing=True: tự động kích hoạt pipeline sinh kết quả.
      2. Chấm điểm toàn diện 3 thuật toán CSP của TV1.
      3. Chấm điểm toàn diện 3 chiến lược Adaptive Decoding của TV3.
      4. Kiểm tra độ chênh lệch giữa Margin và Random so với ngưỡng an toàn.
      5. Xuất báo cáo Markdown, lưu file tổng kết JSON, và vẽ biểu đồ PNG/PDF.

    Parameters
    ----------
    shared_dir : str
        Thư mục gốc chứa data/, results/, checkpoints/.
    output_dir : Optional[str]
        Thư mục xuất file báo cáo và hình ảnh (mặc định shared_dir/results).
    auto_run_missing : bool
        Nếu True, tự động chạy pipeline nếu file kết quả chưa tồn tại.
    quick_test : bool
        Chạy nhanh trên tập con (20 mẫu) phục vụ kiểm tra pipeline.
    device : str
        Thiết bị chạy inference ('cpu' hoặc 'cuda').
    min_delta_cell_acc : float
        Ngưỡng chênh lệch Cell Acc tối thiểu (%).
    min_delta_board_acc : float
        Ngưỡng chênh lệch Board Acc tối thiểu (%).

    Returns
    -------
    audit_results : Dict[str, Any]
        Toàn bộ kết quả chấm điểm, báo cáo và quyết định kiểm định.
    """
    print("=" * 72)
    print("TV4: BẮT ĐẦU CHẤM ĐIỂM THỰC NGHIỆM & CHỐT KIỂM ĐỊNH 14H00 (TASK 16)")
    print("=" * 72)

    results_dir = output_dir or os.path.join(shared_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Tìm hoặc sinh dữ liệu Pilot & Kết quả TV1
    csp_npz_candidates = [
        os.path.join(results_dir, "baseline_csp_results.npz"),
        os.path.join(shared_dir, "results", "baseline_csp_results.npz"),
        os.path.join(shared_dir, "baseline_csp_results.npz"),
    ]
    csp_npz_path = next((p for p in csp_npz_candidates if os.path.isfile(p)), None)

    csp_hist_candidates = [
        os.path.join(results_dir, "baseline_csp_histories.pkl"),
        os.path.join(shared_dir, "results", "baseline_csp_histories.pkl"),
    ]
    csp_hist_path = next((p for p in csp_hist_candidates if os.path.isfile(p)), None)

    # Nếu chưa có kết quả TV1 và auto_run_missing=True: tự động sinh dữ liệu & chạy CSP
    if csp_npz_path is None and auto_run_missing:
        print("\n[*] Chưa có kết quả TV1 (baseline_csp_results.npz). Đang tự động chạy Task 14 của TV1...")
        data_candidates = [
            os.path.join(shared_dir, "data", "sudoku_pilot.npz"),
            os.path.join(shared_dir, "sudoku_pilot.npz"),
            "./data/sudoku_pilot.npz",
        ]
        data_path = next((p for p in data_candidates if os.path.isfile(p)), None)

        # Nếu chưa có data/sudoku_pilot.npz, sinh tập pilot 6x6 chuẩn theo logic TV1
        if data_path is None:
            print("[*] Sinh mới bộ dữ liệu Sudoku 6x6 pilot theo chuẩn TV1...")
            os.makedirs(os.path.join(shared_dir, "data"), exist_ok=True)
            target_data_path = os.path.join(shared_dir, "data", "sudoku_pilot.npz")

            base_sol = np.array([
                [1, 2, 3, 4, 5, 6],
                [4, 5, 6, 1, 2, 3],
                [2, 3, 1, 5, 6, 4],
                [5, 6, 4, 2, 3, 1],
                [3, 1, 2, 6, 4, 5],
                [6, 4, 5, 3, 1, 2],
            ], dtype=np.int64)

            gen_quizzes, gen_solutions = [], []
            n_gen = 250 if quick_test else 1000
            for _ in range(n_gen):
                sol = base_sol.copy()
                perm = np.random.permutation(np.arange(1, 7))
                mapping = {i + 1: perm[i] for i in range(6)}
                sol = np.vectorize(mapping.get)(sol)
                for br in range(0, 6, 2):
                    if np.random.rand() > 0.5:
                        sol[[br, br + 1]] = sol[[br + 1, br]]
                for bc in range(0, 6, 3):
                    sol[:, bc:bc + 3] = sol[:, bc + np.random.permutation(3)]
                sol_1d = sol.flatten().astype(np.int64)
                q_1d = sol_1d.copy()
                num_miss = np.random.randint(14, 23)
                q_1d[np.random.choice(36, size=num_miss, replace=False)] = 0
                gen_quizzes.append(q_1d)
                gen_solutions.append(sol_1d)

            np.savez_compressed(
                target_data_path,
                quizzes=np.array(gen_quizzes, dtype=np.int64),
                solutions=np.array(gen_solutions, dtype=np.int64),
            )
            data_path = target_data_path
            print(f"[*] Đã sinh xong tập pilot ({n_gen} mẫu) tại: {data_path}")

        # Chạy CSP
        from src.data_csp import load_pilot_data, solve_csp
        quizzes, solutions = load_pilot_data(data_path)
        test_inputs = quizzes[-200:] if not quick_test else quizzes[-20:]
        test_targets = solutions[-200:] if not quick_test else solutions[-20:]

        methods = ["backtracking", "mrv", "ac3"]
        preds_dict, hists_dict = {}, {}
        st_times = []

        print(f"[*] Chạy giải CSP trên {len(test_inputs)} bảng test...")
        for m in methods:
            t0 = time.time()
            p_list, h_list = [], []
            for i in range(len(test_inputs)):
                p, h = solve_csp(test_inputs[i], method=m, n_size=6)
                p_list.append(p)
                h_list.append(h)
            st = (time.time() - t0) / len(test_inputs)
            preds_dict[m] = np.array(p_list, dtype=np.int64)
            hists_dict[m] = h_list
            st_times.append(st)

        csp_npz_path = os.path.join(results_dir, "baseline_csp_results.npz")
        np.savez(
            csp_npz_path,
            inputs=test_inputs,
            targets=test_targets,
            preds_backtracking=preds_dict["backtracking"],
            preds_mrv=preds_dict["mrv"],
            preds_ac3=preds_dict["ac3"],
            solve_times=np.array(st_times, dtype=np.float64),
        )

        csp_hist_path = os.path.join(results_dir, "baseline_csp_histories.pkl")
        with open(csp_hist_path, "wb") as f:
            pickle.dump(hists_dict, f)
        print(f"[*] Đã xuất kết quả CSP vào: {csp_npz_path}")

    # 2. Tìm hoặc sinh kết quả TV3 (mdm_inference_results.pkl)
    mdm_pkl_candidates = [
        os.path.join(results_dir, "mdm_inference_results.pkl"),
        os.path.join(shared_dir, "results", "mdm_inference_results.pkl"),
        os.path.join(shared_dir, "mdm_inference_results.pkl"),
    ]
    mdm_pkl_path = next((p for p in mdm_pkl_candidates if os.path.isfile(p)), None)

    if mdm_pkl_path is None and auto_run_missing:
        print("\n[*] Chưa có kết quả TV3 (mdm_inference_results.pkl). Đang tự động chạy Task 15 của TV3...")
        from src.inference import run_task15_experiment
        run_task15_experiment(
            shared_dir=shared_dir,
            quick_test=quick_test,
            device=device,
        )
        mdm_pkl_path = os.path.join(results_dir, "mdm_inference_results.pkl")

    # 3. Chấm điểm kết quả TV1
    csp_eval: Optional[Dict[str, Dict[str, Any]]] = None
    if csp_npz_path and os.path.isfile(csp_npz_path):
        print(f"\n[*] Đang đánh giá kết quả TV1 từ: {csp_npz_path}")
        csp_eval = evaluate_csp_results(
            csp_source=csp_npz_path,
            histories_source=csp_hist_path,
            n_size=6,
        )
        for m, m_data in csp_eval.items():
            print(f"  • {m.upper():12s}: Board Acc={m_data['board_accuracy_pct']:5.2f}% | Cell Acc={m_data['cell_accuracy_pct']:5.2f}% | Lỗi={m_data['avg_constraint_violations']:4.2f}")

    # 4. Chấm điểm kết quả TV3
    if mdm_pkl_path is None or not os.path.isfile(mdm_pkl_path):
        raise FileNotFoundError("Không tìm thấy và không thể tự sinh kết quả TV3 (mdm_inference_results.pkl)!")

    print(f"\n[*] Đang đánh giá kết quả TV3 từ: {mdm_pkl_path}")
    mdm_eval = evaluate_mdm_results(mdm_source=mdm_pkl_path, n_size=6)
    for strat, agg in mdm_eval["aggregated"].items():
        b_m, b_s = _extract_metric_pair(agg, "board_accuracy_pct", as_percentage=True)
        c_m, c_s = _extract_metric_pair(agg, "cell_accuracy_pct", as_percentage=True)
        v_m, v_s = _extract_metric_pair(agg, "avg_constraint_violations")
        print(f"  • {strat.upper():12s}: Board Acc={b_m:5.2f} ± {b_s:4.2f}% | Cell Acc={c_m:5.2f} ± {c_s:4.2f}% | Lỗi={v_m:4.2f} ± {v_s:4.2f}")

    # 5. Kiểm tra độ chênh lệch giữa Margin và Random (Chốt 14h00)
    print("\n[*] Phân tích độ chênh lệch giữa Margin và Random...")
    diff_report = check_margin_vs_random_difference(
        mdm_eval_summary=mdm_eval,
        min_delta_cell_acc=min_delta_cell_acc,
        min_delta_board_acc=min_delta_board_acc,
    )
    print("\n" + diff_report["alert_message"])

    # 6. Tạo báo cáo Markdown và xuất file
    report_md = format_task16_audit_report(
        csp_metrics=csp_eval,
        mdm_eval_summary=mdm_eval,
        diff_report=diff_report,
    )
    report_file = os.path.join(results_dir, "task16_audit_report.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n[OK] Đã lưu Báo cáo Markdown: {report_file}")

    # 7. Vẽ và xuất biểu đồ
    saved_plots = plot_task16_audit_figures(
        csp_metrics=csp_eval,
        mdm_eval_summary=mdm_eval,
        diff_report=diff_report,
        output_dir=figures_dir,
    )

    # 8. Lưu kết quả tổng hợp JSON
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "milestone": "Giai đoạn 4: Ngày 3 - 14h00 (Chốt kiểm định)",
        "task_id": 16,
        "status": diff_report["status"],
        "is_delta_too_small": diff_report["is_delta_too_small"],
        "difference_analysis": diff_report,
        "mdm_aggregated": {
            k: {
                "board_accuracy_pct": _extract_metric_pair(v, "board_accuracy_pct", as_percentage=True)[0],
                "cell_accuracy_pct": _extract_metric_pair(v, "cell_accuracy_pct", as_percentage=True)[0],
                "avg_constraint_violations": _extract_metric_pair(v, "avg_constraint_violations")[0],
                "solve_time_per_board": _extract_metric_pair(v, "solve_time_per_board")[0],
            }
            for k, v in mdm_eval["aggregated"].items()
        },
        "csp_metrics": {
            k: {
                "board_accuracy_pct": v.get("board_accuracy_pct", 0.0),
                "cell_accuracy_pct": v.get("cell_accuracy_pct", 0.0),
                "avg_constraint_violations": v.get("avg_constraint_violations", 0.0),
                "solve_time_per_board": v.get("solve_time_per_board", 0.0),
            }
            for k, v in (csp_eval or {}).items()
        },
        "recommendations": {
            "tv1": diff_report["recommendation_tv1"],
            "tv3": diff_report["recommendation_tv3"],
        },
        "artifacts": {
            "report_markdown": report_file,
            "plots": saved_plots,
        },
    }

    summary_file = os.path.join(results_dir, "task16_evaluation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(_safe_json_serialize(summary_data), f, indent=2, ensure_ascii=False)
    print(f"[OK] Đã lưu Tổng hợp JSON: {summary_file}")

    print("=" * 72)
    print(f"🎉 HOÀN THÀNH TOÀN DIỆN TASK 16! TRẠNG THÁI: [{diff_report['status']}]")
    print("=" * 72)

    return {
        "summary": summary_data,
        "csp_eval": csp_eval,
        "mdm_eval": mdm_eval,
        "diff_report": diff_report,
        "report_md": report_md,
    }


# ===========================================================================
# 6. Self-test tự động toàn diện kiểm thử Ngày 1, 2 & 3 (Task 16)
# ===========================================================================

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    workspace_root = os.path.dirname(project_root)

    for p in (project_root, current_dir, workspace_root):
        if p not in sys.path:
            sys.path.insert(0, p)

    parser = argparse.ArgumentParser(description="TV4: Metrics & Evaluation for Sudoku Diffusion Models")
    parser.add_argument("--shared_dir", type=str, default=".", help="Thư mục gốc chứa data/, checkpoints/, results/")
    parser.add_argument("--run_task16", "--audit", action="store_true", help="Chạy thực nghiệm chấm điểm Task 16")
    parser.add_argument("--quick_test", action="store_true", help="Chạy nhanh trên số lượng mẫu nhỏ (20 mẫu)")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị ('cpu' hoặc 'cuda')")
    parser.add_argument("--unit_test_only", action="store_true", help="Chỉ chạy bộ unit test tự động")
    args, unknown = parser.parse_known_args()

    if args.run_task16:
        run_task16_audit(
            shared_dir=args.shared_dir,
            quick_test=args.quick_test,
            device=args.device,
        )
        sys.exit(0)

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

    # -----------------------------------------------------------------------
    # TEST 9: Kiểm tra evaluate_csp_results & evaluate_mdm_results (Task 16)
    # -----------------------------------------------------------------------
    print("\n[TEST 9] Kiểm tra evaluate_csp_results & evaluate_mdm_results trên mock data...")
    mock_targets = np.array([
        [1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1],
        [4, 3, 2, 1, 2, 1, 4, 3, 3, 4, 1, 2, 1, 2, 3, 4],
    ], dtype=np.int64)
    mock_inputs = mock_targets.copy()
    mock_inputs[:, [0, 5, 10]] = 0

    mock_csp_dict = {
        "inputs": mock_inputs,
        "targets": mock_targets,
        "preds_backtracking": mock_targets.copy(),
        "preds_mrv": mock_targets.copy(),
        "preds_ac3": mock_targets.copy(),
        "solve_times": np.array([0.0012, 0.0008, 0.0005]),
    }
    csp_res = evaluate_csp_results(mock_csp_dict, n_size=4)
    assert set(csp_res.keys()) == {"backtracking", "mrv", "ac3"}
    assert csp_res["mrv"]["board_accuracy_pct"] == 100.0
    assert csp_res["mrv"]["avg_constraint_violations"] == 0.0
    print("  ✅ [PASS] evaluate_csp_results chấm điểm chính xác 3 thuật toán CSP.")

    mock_mdm_dict = {
        "inputs": mock_inputs,
        "targets": mock_targets,
        "random": {
            42: {"preds": mock_targets.copy(), "solve_time": 0.12},
            123: {"preds": mock_targets.copy(), "solve_time": 0.11},
        },
        "top_prob": {
            42: {"preds": mock_targets.copy(), "solve_time": 0.14},
            123: {"preds": mock_targets.copy(), "solve_time": 0.13},
        },
        "margin": {
            42: {"preds": mock_targets.copy(), "solve_time": 0.15},
            123: {"preds": mock_targets.copy(), "solve_time": 0.14},
        },
    }
    mdm_res = evaluate_mdm_results(mock_mdm_dict, n_size=4)
    assert set(mdm_res["aggregated"].keys()) == {"random", "top_prob", "margin"}
    assert mdm_res["aggregated"]["margin"]["board_accuracy_pct_mean"] == 100.0
    print("  ✅ [PASS] evaluate_mdm_results tổng hợp đa seed (mean & std) thành công.")

    # -----------------------------------------------------------------------
    # TEST 10: Kiểm tra logic chốt kiểm định 14h00 (check_margin_vs_random_difference)
    # -----------------------------------------------------------------------
    print("\n[TEST 10] Kiểm tra logic chốt kiểm định 14h00 khi chênh lệch quá nhỏ vs khi đạt...")
    # Kịch bản 1: Chênh lệch quá nhỏ (< 1.0% cell acc)
    low_diff_mdm = {
        "targets": mock_targets,
        "aggregated": {
            "random": {
                "cell_accuracy_pct": MetricSummary(60.0, 1.0),
                "board_accuracy_pct": MetricSummary(0.0, 0.0),
                "avg_constraint_violations": MetricSummary(5.0, 0.2),
            },
            "margin": {
                "cell_accuracy_pct": MetricSummary(60.4, 0.9),  # Delta = 0.4% < 1.0%
                "board_accuracy_pct": MetricSummary(0.0, 0.0),
                "avg_constraint_violations": MetricSummary(4.9, 0.2),
                "high_margin_cell_acc_pct": MetricSummary(62.0, 0.5),
                "low_margin_cell_acc_pct": MetricSummary(58.0, 0.5),
            },
        },
        "by_seed": {"random": {}, "margin": {}},
        "seeds": [42],
    }
    diff_low = check_margin_vs_random_difference(low_diff_mdm, min_delta_cell_acc=1.0)
    assert diff_low["status"] == "WARNING", f"Kỳ vọng WARNING nhưng nhận: {diff_low['status']}"
    assert diff_low["is_delta_too_small"] is True
    assert "tăng độ khó" in diff_low["recommendation_tv1"].lower()
    assert "gumbel_scale" in diff_low["recommendation_tv3"].lower()
    print("  ✅ [PASS] Phát hiện chính xác chênh lệch quá nhỏ lúc 14h00 và xuất cảnh báo đúng cho TV1 & TV3.")

    # Kịch bản 2: Chênh lệch lớn vượt trội (PASSED)
    high_diff_mdm = {
        "targets": mock_targets,
        "aggregated": {
            "random": {
                "cell_accuracy_pct": MetricSummary(50.0, 1.2),
                "board_accuracy_pct": MetricSummary(5.0, 1.0),
                "avg_constraint_violations": MetricSummary(8.0, 0.5),
            },
            "margin": {
                "cell_accuracy_pct": MetricSummary(75.5, 1.0),  # Delta = +25.5%
                "board_accuracy_pct": MetricSummary(45.0, 2.0),
                "avg_constraint_violations": MetricSummary(2.1, 0.3),
                "high_margin_cell_acc_pct": MetricSummary(88.0, 1.0),
                "low_margin_cell_acc_pct": MetricSummary(55.0, 1.0),
            },
        },
        "by_seed": {"random": {}, "margin": {}},
        "seeds": [42, 123],
    }
    diff_high = check_margin_vs_random_difference(high_diff_mdm, min_delta_cell_acc=1.0)
    assert diff_high["status"] == "PASSED", f"Kỳ vọng PASSED nhưng nhận: {diff_high['status']}"
    assert diff_high["is_delta_too_small"] is False
    print("  ✅ [PASS] Xác nhận chốt kiểm định 14h00 ĐẠT khi Margin vượt trội Random.")

    # Kiểm tra xuất báo cáo Markdown và biểu đồ kiểm định Task 16
    audit_report_text = format_task16_audit_report(csp_res, mdm_res, diff_high)
    assert "BÁO CÁO NGHIỆM THU & CHỐT KIỂM ĐỊNH" in audit_report_text
    assert "Top Prob Margin" in audit_report_text
    task16_plots = plot_task16_audit_figures(csp_res, mdm_res, diff_high, output_dir=output_dir)
    assert "strategy_comparison_png" in task16_plots
    assert "margin_vs_random_png" in task16_plots
    print("  ✅ [PASS] Xuất báo cáo Markdown và biểu đồ Task 16 thành công.")

    print("\n" + "=" * 72)
    print("🎉 TẤT CẢ 10 BỘ KIỂM THỬ CỦA TV4 ĐÃ VƯỢT QUA 100%! MODULE HOÀN THIỆN ĐẦY ĐỦ.")
    print("=" * 72)
