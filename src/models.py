"""
src/models.py – TV2: Kiến trúc TinyMDM (Pre-LN) & Loss Function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyMDM(nn.Module):
    """Mô hình Masked Diffusion dùng Bidirectional Self-Attention (Pre-LN)."""

    def __init__(
        self,
        vocab_size: int = 7,
        seq_len: int = 36,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        # x_t: (Batch, L) chứa 0..N (0 là [MASK]) -> logits: (Batch, L, vocab_size)
        batch_size, length = x_t.size()
        h = self.token_embedding(x_t) + self.pos_embedding[:, :length, :]
        h = self.transformer(h)
        h = self.final_norm(h)
        return self.head(h)


def compute_mdm_loss(
    logits: torch.Tensor, x_0: torch.Tensor, x_t: torch.Tensor
) -> torch.Tensor:
    """Tính Cross-Entropy chỉ trên các vị trí bị che (x_t == 0) và chặn token [MASK]."""
    masked_pos = (x_t == 0)
    if masked_pos.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits = logits.clone()
    logits[:, :, 0] = -1e9
    return F.cross_entropy(logits[masked_pos], x_0[masked_pos])


def load_pretrained_mdm(
    ckpt_path: str,
    device: str = "cpu",
    vocab_size: int = 7,
    seq_len: int = 36,
) -> TinyMDM:
    """Hàm tiện ích nạp checkpoint mdm_pilot_best.pt cho TV3 và TV5."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        v_size = ckpt.get("vocab_size", vocab_size)
        s_len = ckpt.get("seq_len", seq_len)
        state_dict = ckpt["model_state_dict"]
    else:
        v_size, s_len = vocab_size, seq_len
        state_dict = ckpt

    model = TinyMDM(vocab_size=v_size, seq_len=s_len).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model