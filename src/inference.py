"""
src/inference.py – TV3: Adaptive Decoding Strategies
====================================================
File thuộc quyền quản lý của Thành viên 3.
TV5 chỉ tạo khung chữ ký hàm theo Data Contract, KHÔNG viết logic bên trong.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict


def adaptive_decode(
    model: nn.Module,
    quiz_1d: np.ndarray,
    strategy: str = "margin",
    steps: int = 50,
    gumbel_scale: float = 0.5,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[Dict]]:
    """Giải mã thích ứng bảng Sudoku bằng Masked Diffusion Model.

    Parameters
    ----------
    model : nn.Module
        Model MDM đã huấn luyện (hoặc DummyMDM để test).
    quiz_1d : np.ndarray, shape (L,), dtype int64
        Bảng đề 1D (0 = [MASK]).
    strategy : str
        Chiến lược giải mã: 'random', 'top_prob', 'margin'.
    steps : int
        Số bước giải mã tối đa.
    gumbel_scale : float
        Hệ số nhiễu Gumbel (chỉ dùng cho strategy='margin').
    device : str
        Device chạy model ('cpu' hoặc 'cuda').

    Returns
    -------
    pred_board_1d : np.ndarray, shape (L,), dtype int64
        Bảng kết quả dự đoán.
    step_history : List[Dict]
        Lịch sử từng bước giải, mỗi phần tử có 5 khóa:
        {'step', 'unmasked_indices', 'values', 'margins', 'board_state'}.
    """
    raise NotImplementedError("TV3 sẽ triển khai hàm này.")
