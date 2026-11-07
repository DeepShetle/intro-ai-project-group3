"""
src/metrics.py – TV4: Đánh giá & Metrics
=========================================
File thuộc quyền quản lý của Thành viên 4.
TV5 chỉ tạo khung chữ ký hàm theo Data Contract, KHÔNG viết logic bên trong.
"""

import numpy as np
from typing import List, Dict


def evaluate_predictions(
    preds_2d: np.ndarray,
    targets_2d: np.ndarray,
    histories: List[List[Dict]],
    n_size: int = 9,
    box_r: int = 3,
    box_c: int = 3,
) -> Dict:
    """Đánh giá tổng hợp kết quả dự đoán Sudoku.

    Parameters
    ----------
    preds_2d : np.ndarray, shape (N, L), dtype int64
        Mảng các bảng dự đoán.
    targets_2d : np.ndarray, shape (N, L), dtype int64
        Mảng các bảng đáp án.
    histories : List[List[Dict]]
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
        Dictionary chứa các metric đánh giá (accuracy, violation_rate, …).
    """
    raise NotImplementedError("TV4 sẽ triển khai hàm này.")


def count_sudoku_violations(
    board_1d: np.ndarray,
    n_size: int = 9,
    box_r: int = 3,
    box_c: int = 3,
) -> int:
    """Đếm số vi phạm ràng buộc Sudoku trên một bảng.

    Parameters
    ----------
    board_1d : np.ndarray, shape (L,), dtype int64
        Bảng Sudoku 1D.
    n_size : int
        Kích thước bảng.
    box_r : int
        Số hàng trong mỗi box con.
    box_c : int
        Số cột trong mỗi box con.

    Returns
    -------
    violations : int
        Tổng số vi phạm (trùng lặp) trong hàng, cột, box.
    """
    raise NotImplementedError("TV4 sẽ triển khai hàm này.")
