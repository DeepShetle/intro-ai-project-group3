"""
src/models.py – TV2: Kiến trúc TinyMDM & TinyARM
===============================================
File thuộc quyền quản lý của Thành viên 2.
TV5 chỉ tạo khung chữ ký hàm theo Data Contract, KHÔNG viết logic bên trong.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- THÊM MỚI (phục vụ tính Loss)


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


# =====================================================================
# MÔ HÌNH BASELINE AUTOREGRESSIVE (TV2 - GIAI ĐOẠN 4)
# =====================================================================
class TinyARM(nn.Module):
    """Mô hình Autoregressive nhỏ gọn (TinyARM) cho Sudoku (Baseline trái-sang-phải).

    Sinh các ô lần lượt từ vị trí 0 đến 80 bằng Causal Attention Mask.
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

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Transformer Encoder Layer kết hợp causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass với Causal Mask (chỉ chú ý về bên trái)."""
        batch_size, length = x.size()
        device = x.device

        # Causal mask hình tam giác: True nghĩa là bị che không được nhìn
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length).to(device)

        h = self.token_embedding(x) + self.pos_embedding[:, :length, :]
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        logits = self.head(h)
        return logits


# =====================================================================
# HÀM MỚI THÊM CHO GIAI ĐOẠN 3 (TV2)
# =====================================================================
def compute_mdm_loss(
    logits: torch.Tensor, x_0: torch.Tensor, x_t: torch.Tensor
) -> torch.Tensor:
    """Tính Cross-Entropy Loss cho Masked Diffusion Model.

    Chỉ tính loss tại các vị trí ô bị mask (nơi x_t == 0).
    """
    masked_pos = x_t == 0

    if masked_pos.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    masked_logits = logits[masked_pos]
    masked_targets = x_0[masked_pos]

    loss = F.cross_entropy(masked_logits, masked_targets)
    return loss


if __name__ == "__main__":
    # --- KIỂM THỬ KHUNG MẠNG TRỰC TIẾP TRÊN LOCAL ---
    print("--- KIỂM THỬ TINYMDM ---")
    model = TinyMDM()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Tổng số tham số mô hình TinyMDM: {num_params:,}")

    mock_x_t = torch.randint(0, 10, (4, 81))
    logits = model(mock_x_t)
    print(f"✅ Kích thước Input (x_t): {mock_x_t.shape}")
    print(f"✅ Kích thước Output (logits): {logits.shape}")

    # --- KIỂM THỬ HÀM LOSS ---
    mock_x_0 = torch.randint(1, 10, (4, 81))
    mock_x_t[:, :30] = 0  # Giả lập che 30 ô
    logits = model(mock_x_t)
    loss = compute_mdm_loss(logits, mock_x_0, mock_x_t)
    print(f"✅ Giá trị Loss tính thử: {loss.item():.4f}")

    # --- KIỂM THỬ TINYARM ---
    print("\n--- KIỂM THỬ TINYARM ---")
    arm_model = TinyARM()
    arm_params = sum(p.numel() for p in arm_model.parameters() if p.requires_grad)
    print(f"✅ Tổng số tham số mô hình TinyARM: {arm_params:,}")

    mock_arm_in = torch.randint(0, 10, (4, 81))
    arm_logits = arm_model(mock_arm_in)
    print(f"✅ Kích thước Output TinyARM (logits): {arm_logits.shape}")