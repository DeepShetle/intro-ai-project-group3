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
    data = np.load(npz_path)
    quizzes = data['quizzes'].astype(np.int64)
    solutions = data['solutions'].astype(np.int64)
    return quizzes, solutions


def _get_box_dims(n_size: int) -> Tuple[int, int]:
    """Lấy kích thước hàng và cột của khối con (subgrid box)."""
    if n_size == 4:
        return 2, 2
    elif n_size == 6:
        return 2, 3
    elif n_size == 9:
        return 3, 3
    else:
        root = int(np.sqrt(n_size))
        return root, root


def _get_candidates(board_2d: np.ndarray, r: int, c: int, n_size: int, box_r: int, box_c: int) -> set:
    """Trả về tập hợp các số hợp lệ (1..N) có thể điền vào ô (r, c)."""
    if board_2d[r, c] != 0:
        return set()
    
    # Ràng buộc hàng và cột
    used = set(board_2d[r, :]) | set(board_2d[:, c])
    
    # Ràng buộc khối con
    br, bc = (r // box_r) * box_r, (c // box_c) * box_c
    used |= set(board_2d[br:br + box_r, bc:bc + box_c].flatten())
    
    return set(range(1, n_size + 1)) - used


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
    box_r, box_c = _get_box_dims(n_size)
    board = quiz_1d.copy().astype(np.int64)
    step_history: List[Dict] = []
    step_counter = [0]

    use_simple_backtrack = method in ["backtrack", "backtracking"]

    def select_variable(b_2d: np.ndarray):
        """Chọn ô trống tiếp theo dựa theo heuristic."""
        if use_simple_backtrack:
            for r in range(n_size):
                for c in range(n_size):
                    if b_2d[r, c] == 0:
                        cands = _get_candidates(b_2d, r, c, n_size, box_r, box_c)
                        return (r, c), cands
            return None, set()

        # Heuristic MRV: chọn ô có ít giá trị hợp lệ còn lại nhất (tương đương Margin cao nhất)
        best_pos = None
        min_cands_count = n_size + 2
        best_cands = set()

        for r in range(n_size):
            for c in range(n_size):
                if b_2d[r, c] == 0:
                    cands = _get_candidates(b_2d, r, c, n_size, box_r, box_c)
                    num_c = len(cands)
                    if num_c == 0:
                        return (r, c), set()  # Nhánh cụt (dead-end)
                    if num_c < min_cands_count:
                        min_cands_count = num_c
                        best_pos = (r, c)
                        best_cands = cands
                        if min_cands_count == 1:
                            return best_pos, best_cands
        return best_pos, best_cands

    def backtrack() -> bool:
        b_2d = board.reshape(n_size, n_size)
        pos, candidates = select_variable(b_2d)

        # Hết ô trống -> Đã tìm thấy nghiệm
        if pos is None:
            return True

        if len(candidates) == 0:
            return False

        r, c = pos
        idx = r * n_size + c

        # Duyệt qua các giá trị ứng viên hợp lệ (Forward Checking)
        for val in sorted(list(candidates)):
            board[idx] = val
            step_counter[0] += 1

            margin_proxy = round(1.0 / float(max(1, len(candidates))), 4)

            step_history.append({
                "step": int(step_counter[0]),
                "unmasked_indices": [int(idx)],
                "values": [int(val)],
                "margins": [float(margin_proxy)],
                "board_state": board.copy()
            })

            if backtrack():
                return True

            # Quay lui (backtrack) nếu bế tắc
            board[idx] = 0

        return False

    backtrack()
    return board, step_history