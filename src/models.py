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
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model

        # 1. Token Embedding (0: [MASK], 1..9: chữ số Sudoku)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Embedding cho mảng 1D (81 vị trí)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        # 3. Transformer Encoder với chú ý hai chiều (Bidirectional Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Output Projection Head chuyển về phân phối Logits (vocab_size)
        self.head = nn.Linear(d_model, vocab_size)

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
        batch_size, length = x_t.size()

        # Cộng Token Embedding với Positional Embedding
        h = self.token_embedding(x_t) + self.pos_embedding[:, :length, :]

        # Truyền qua Transformer Encoder
        h = self.transformer(h)

        # Trả về Logits kích thước (B, L, vocab_size)
        logits = self.head(h)
        return logits


if __name__ == "__main__":
    # --- KIỂM THỬ KHUNG MẠNG TRỰC TIẾP TRÊN LOCAL ---
    model = TinyMDM()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Tổng số tham số mô hình: {num_params:,}")

    # Giả lập input x_t (Batch=4, L=81)
    mock_x_t = torch.randint(0, 10, (4, 81))
    logits = model(mock_x_t)

    print(f"✅ Kích thước Input (x_t): {mock_x_t.shape}")
    print(f"✅ Kích thước Output (logits): {logits.shape}")