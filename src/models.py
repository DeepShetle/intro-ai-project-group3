"""
src/models.py – TV2: Kiến trúc TinyMDM
=======================================
File thuộc quyền quản lý của Thành viên 2.
TV5 chỉ tạo khung chữ ký hàm theo Data Contract, KHÔNG viết logic bên trong.
"""

import torch
import torch.nn as nn


class TinyMDM(nn.Module):
    """Masked Diffusion Model nhỏ gọn cho Sudoku.

    Parameters
    ----------
    vocab_size : int
        Kích thước vocabulary (N+1, gồm cả token MASK=0).
    seq_len : int
        Chiều dài chuỗi đầu vào (L = n_size^2).
    d_model : int
        Kích thước embedding / hidden dimension.
    """

    def __init__(
        self,
        vocab_size: int = 10,
        seq_len: int = 81,
        d_model: int = 128,
    ):
        super().__init__()
        raise NotImplementedError("TV2 sẽ triển khai kiến trúc model.")

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Forward pass: dự đoán logits cho mỗi vị trí.

        Parameters
        ----------
        x_t : torch.Tensor, shape (B, L), dtype long
            Batch bảng Sudoku đã bị mask.

        Returns
        -------
        logits : torch.Tensor, shape (B, L, vocab_size)
            Logits dự đoán cho từng vị trí.
        """
        raise NotImplementedError("TV2 sẽ triển khai forward pass.")
