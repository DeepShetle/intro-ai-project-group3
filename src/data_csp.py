"""
src/data_csp.py – TV1: Dữ liệu & CSP Solver
=============================================
File thuộc quyền quản lý của Thành viên 1.
TV5 chỉ tạo khung chữ ký hàm theo Data Contract, KHÔNG viết logic bên trong.
"""

from collections import deque
from typing import Dict, List, Set, Tuple
import numpy as np


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


def _get_neighbors(idx: int, n_size: int, box_r: int, box_c: int) -> Set[int]:
    """Lấy tập hợp các chỉ số ô hàng xóm (cùng hàng, cột, khối con) với ô idx."""
    r, c = idx // n_size, idx % n_size
    neighbors = set()

    # Ràng buộc cùng hàng và cùng cột
    for i in range(n_size):
        if i != c:
            neighbors.add(r * n_size + i)
        if i != r:
            neighbors.add(i * n_size + c)

    # Ràng buộc cùng khối con (box)
    br, bc = (r // box_r) * box_r, (c // box_c) * box_c
    for row in range(br, br + box_r):
        for col in range(bc, bc + box_c):
            neigh_idx = row * n_size + col
            if neigh_idx != idx:
                neighbors.add(neigh_idx)

    return neighbors


def _revise(xi: int, xj: int, domains: Dict[int, Set[int]]) -> bool:
    """Hàm revise loại bỏ giá trị x trong Domain(xi) nếu xj không còn giá trị thỏa x != y."""
    revised = False
    to_remove = set()
    for x in domains[xi]:
        if domains[xj] == {x}:
            to_remove.add(x)
            revised = True
    domains[xi] -= to_remove
    return revised


def _ac3(
    domains: Dict[int, Set[int]],
    neighbors_map: Dict[int, Set[int]],
    queue: deque = None,
) -> bool:
    """Thuật toán duy trì hàng đợi cung Arc Queue để lọc/rút gọn miền giá trị."""
    if queue is None:
        queue = deque()
        for xi in domains:
            for xj in neighbors_map[xi]:
                queue.append((xi, xj))

    while queue:
        xi, xj = queue.popleft()
        if _revise(xi, xj, domains):
            if len(domains[xi]) == 0:
                return False  # Xuất hiện miền giá trị rỗng -> xung đột/ngõ cụt
            for xk in neighbors_map[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True


def solve_csp(
    quiz_1d: np.ndarray,
    method: str = "mrv",
    n_size: int = 6,
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
    seq_len = n_size * n_size
    step_history: List[Dict] = []
    step_counter = [0]

    # Bản đồ hàng xóm cho từng vị trí
    neighbors_map = {
        i: _get_neighbors(i, n_size, box_r, box_c) for i in range(seq_len)
    }

    # Khởi tạo miền giá trị (Domain) cho từng ô
    domains: Dict[int, Set[int]] = {}
    for i in range(seq_len):
        if board[i] != 0:
            domains[i] = {int(board[i])}
        else:
            domains[i] = set(range(1, n_size + 1))

    # Tỉa bớt miền giá trị ban đầu dựa theo các ô đã cho sẵn
    for i in range(seq_len):
        if board[i] != 0:
            val = int(board[i])
            for neigh in neighbors_map[i]:
                domains[neigh].discard(val)

    # Nếu chạy AC-3 thuần: Thực hiện Arc Consistency trên toàn bảng trước khi tìm kiếm
    if method == "ac3":
        if not _ac3(domains, neighbors_map):
            return board, step_history

    def select_variable():
        """Chọn biến tiếp theo cần gán."""
        if method in ["backtrack", "backtracking"]:
            for idx in range(seq_len):
                if board[idx] == 0:
                    return idx, sorted(list(domains[idx]))
            return None, []

        # Heuristic MRV (Minimum Remaining Values): Chọn ô có ít ứng viên hợp lệ nhất
        best_idx = None
        min_cands_count = n_size + 2
        for idx in range(seq_len):
            if board[idx] == 0:
                num_c = len(domains[idx])
                if num_c == 0:
                    return idx, []  # Ngõ cụt (dead-end)
                if num_c < min_cands_count:
                    min_cands_count = num_c
                    best_idx = idx
                    if min_cands_count == 1:
                        break
        if best_idx is None:
            return None, []
        return best_idx, sorted(list(domains[best_idx]))

    def backtrack() -> bool:
        idx, candidates = select_variable()

        # Không còn ô trống nào -> Đã tìm ra lời giải hợp lệ
        if idx is None:
            return True

        if len(candidates) == 0:
            return False

        for val in candidates:
            # Lưu lại trạng thái domains để khôi phục khi quay lui
            saved_domains = {k: v.copy() for k, v in domains.items()}

            board[idx] = val
            domains[idx] = {val}
            step_counter[0] += 1

            margin_proxy = round(1.0 / float(max(1, len(candidates))), 4)
            step_history.append({
                "step": int(step_counter[0]),
                "unmasked_indices": [int(idx)],
                "values": [int(val)],
                "margins": [float(margin_proxy)],
                "board_state": board.copy(),
            })

            consistent = True
            if method == "ac3":
                # Đẩy các cung liên quan đến ô vừa gán vào hàng đợi để lan truyền ràng buộc
                arc_queue = deque([(neigh, idx) for neigh in neighbors_map[idx]])
                consistent = _ac3(domains, neighbors_map, arc_queue)
            else:
                # Forward Checking cơ bản: Loại bỏ giá trị val khỏi miền giá trị các ô hàng xóm
                for neigh in neighbors_map[idx]:
                    domains[neigh].discard(val)
                    if len(domains[neigh]) == 0 and board[neigh] == 0:
                        consistent = False
                        break

            if consistent and backtrack():
                return True

            # Quay lui (backtrack) nếu bế tắc
            board[idx] = 0
            domains.update(saved_domains)

        return False

    backtrack()
    return board, step_history