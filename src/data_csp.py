"""
src/data_csp.py – TV1: Dữ liệu & CSP Solver
=============================================
File thuộc quyền quản lý của Thành viên 1.
TV5 chỉ tạo khung chữ ký hàm theo Data Contract, KHÔNG viết logic bên trong.
"""

import numpy as np
from typing import Tuple, List, Dict


def load_pilot_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc file .npz chứa dữ liệu Sudoku pilot.

    Parameters
    ----------
    npz_path : str
        Đường dẫn tới file .npz chứa 'quizzes' và 'solutions'.

    Returns
    -------
    quizzes : np.ndarray, shape (N, L), dtype int64
        Mảng các bảng đề (0 = [MASK]).
    solutions : np.ndarray, shape (N, L), dtype int64
        Mảng các bảng đáp án tương ứng.
    """
    raise NotImplementedError("TV1 sẽ triển khai hàm này.")


def solve_csp(
    quiz_1d: np.ndarray,
    method: str = "mrv",
    n_size: int = 9,
) -> Tuple[np.ndarray, List[Dict]]:
    """Giải một bảng Sudoku bằng thuật toán CSP cổ điển.

    Parameters
    ----------
    quiz_1d : np.ndarray, shape (L,), dtype int64
        Bảng đề 1D (0 = [MASK]).
    method : str
        Phương pháp CSP: 'backtrack', 'mrv', 'ac3'.
    n_size : int
        Kích thước bảng (4, 6, hoặc 9).

    Returns
    -------
    pred_board_1d : np.ndarray, shape (L,), dtype int64
        Bảng kết quả dự đoán.
    step_history : List[Dict]
        Lịch sử từng bước giải, mỗi phần tử có 5 khóa:
        {'step', 'unmasked_indices', 'values', 'margins', 'board_state'}.
    """
    raise NotImplementedError("TV1 sẽ triển khai hàm này.")
