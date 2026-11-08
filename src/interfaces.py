"""
src/interfaces.py – TV5: Integration Interfaces & Mock Data
===========================================================
Cung cấp:
  - DummyMDM        : Model giả lập để test pipeline khi chưa có TinyMDM thật.
  - generate_mock_data          : Sinh dữ liệu (quizzes, solutions) giả lập.
  - generate_mock_step_history  : Sinh step_history giả lập đúng Data Contract.
  - validate_step_history       : Kiểm tra tự động step_history đúng chuẩn.
"""

from __future__ import annotations

import numpy as np
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
    _BaseModule = nn.Module
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False
    _BaseModule = object
from typing import List, Dict, Tuple, Any


# ---------------------------------------------------------------------------
# 1. DummyMDM – Model giả lập
# ---------------------------------------------------------------------------

class DummyMDM(_BaseModule):
    """Masked Diffusion Model giả lập trả logits ngẫu nhiên.

    Dùng để các thành viên (TV3, TV4, TV5) test pipeline mà không cần
    model đã huấn luyện thật.

    Parameters
    ----------
    vocab_size : int
        Kích thước vocabulary (mặc định 10 = 0..9 cho Sudoku 9×9).
    """

    def __init__(self, vocab_size: int = 10):
        if _HAS_TORCH:
            super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x_t: Any) -> Any:
        """Trả logits ngẫu nhiên có shape (B, L, vocab_size).

        Parameters
        ----------
        x_t : torch.Tensor, shape (B, L), dtype long
            Batch bảng Sudoku (có thể chứa MASK = 0).

        Returns
        -------
        logits : torch.Tensor, shape (B, L, vocab_size)
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch chưa được cài đặt. Vui lòng cài đặt: pip install torch")
        B, L = x_t.shape
        return torch.randn(B, L, self.vocab_size, device=x_t.device)


# ---------------------------------------------------------------------------
# 2. generate_mock_data – Sinh dữ liệu giả lập
# ---------------------------------------------------------------------------

def generate_mock_data(
    num_samples: int = 10,
    n_size: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sinh dữ liệu Sudoku giả lập (quizzes, solutions).

    Quy ước Vocabulary:
      - 0 = [MASK]  (ô trống trong quiz)
      - 1..N = giá trị điền vào bảng

    **Lưu ý:** Solutions được sinh ngẫu nhiên, KHÔNG đảm bảo hợp lệ
    theo luật Sudoku. Chỉ dùng để test cấu trúc pipeline.

    Parameters
    ----------
    num_samples : int
        Số lượng bảng cần sinh.
    n_size : int
        Kích thước bảng (4, 6, hoặc 9).

    Returns
    -------
    quizzes : np.ndarray, shape (num_samples, L), dtype int64
        Mảng các bảng đề (chứa 0 tại các ô trống).
    solutions : np.ndarray, shape (num_samples, L), dtype int64
        Mảng các bảng đáp án (chỉ chứa giá trị 1..N).
    """
    L = n_size * n_size  # 16, 36, hoặc 81

    # Solutions: giá trị ngẫu nhiên trong [1, N]
    solutions = np.random.randint(1, n_size + 1, size=(num_samples, L)).astype(np.int64)

    # Quizzes: copy solution rồi mask khoảng 40-70% ô
    quizzes = solutions.copy()
    for i in range(num_samples):
        num_masks = np.random.randint(int(L * 0.4), int(L * 0.7) + 1)
        mask_indices = np.random.choice(L, size=num_masks, replace=False)
        quizzes[i, mask_indices] = 0  # 0 = [MASK]

    return quizzes, solutions


# ---------------------------------------------------------------------------
# 3. generate_mock_step_history – Sinh step_history giả lập
# ---------------------------------------------------------------------------

def generate_mock_step_history(
    quiz_1d: np.ndarray,
    n_size: int = 9,
) -> List[Dict]:
    """Sinh một step_history giả lập tuân thủ đúng Data Contract.

    Mỗi bước mở 1 ô MASK theo thứ tự ngẫu nhiên, gán giá trị ngẫu nhiên
    trong [1, N] và margin ngẫu nhiên trong (0, 1).

    Data Contract yêu cầu mỗi phần tử trong step_history có đúng 5 khóa:
      - step            : int         – số thứ tự bước (bắt đầu từ 1).
      - unmasked_indices : list[int]   – chỉ số các ô được mở tại bước này.
      - values           : list[int]   – giá trị gán cho các ô được mở.
      - margins          : list[float] – độ tin cậy / margin tương ứng.
      - board_state      : np.ndarray  – trạng thái bảng SAU bước này (shape (L,)).

    Parameters
    ----------
    quiz_1d : np.ndarray, shape (L,), dtype int64
        Bảng đề 1D (0 = [MASK]).
    n_size : int
        Kích thước bảng (4, 6, hoặc 9).

    Returns
    -------
    step_history : List[Dict]
        Danh sách các bước giải giả lập.
    """
    L = len(quiz_1d)
    board = quiz_1d.copy().astype(np.int64)

    # Tìm tất cả các vị trí MASK
    mask_positions = np.where(board == 0)[0].tolist()
    np.random.shuffle(mask_positions)

    step_history: List[Dict] = []

    for step_idx, pos in enumerate(mask_positions, start=1):
        value = int(np.random.randint(1, n_size + 1))
        margin = float(np.random.uniform(0.0, 1.0))

        # Mở ô
        board[pos] = value

        step_history.append({
            "step": step_idx,
            "unmasked_indices": [int(pos)],
            "values": [value],
            "margins": [margin],
            "board_state": board.copy(),
        })

    return step_history


# ---------------------------------------------------------------------------
# 4. validate_step_history – Kiểm tra Data Contract
# ---------------------------------------------------------------------------

def validate_step_history(
    step_history: list,
    seq_len: int = 81,
) -> bool:
    """Kiểm tra tự động xem step_history có đúng Data Contract không.

    Kiểm tra:
      1. step_history là list, mỗi phần tử là dict.
      2. Mỗi dict có đúng 5 khóa bắt buộc.
      3. Kiểu dữ liệu của từng khóa đúng quy định.
      4. Số thứ tự step liên tục bắt đầu từ 1.
      5. Độ dài unmasked_indices == values == margins.
      6. board_state có shape (seq_len,).

    Parameters
    ----------
    step_history : list
        Danh sách step_history cần kiểm tra.
    seq_len : int
        Chiều dài kỳ vọng của board_state (mặc định 81 cho 9×9).

    Returns
    -------
    is_valid : bool
        True nếu step_history hợp lệ, False nếu không.
    """
    REQUIRED_KEYS = {"step", "unmasked_indices", "values", "margins", "board_state"}

    if not isinstance(step_history, list):
        print("[VALIDATE] ❌ step_history phải là list.")
        return False

    if len(step_history) == 0:
        print("[VALIDATE] ⚠️  step_history rỗng (có thể hợp lệ nếu không có ô MASK).")
        return True

    for i, entry in enumerate(step_history):
        # --- Kiểm tra kiểu dict ---
        if not isinstance(entry, dict):
            print(f"[VALIDATE] ❌ Phần tử [{i}] không phải dict, mà là {type(entry)}.")
            return False

        # --- Kiểm tra đủ 5 khóa ---
        if set(entry.keys()) != REQUIRED_KEYS:
            missing = REQUIRED_KEYS - set(entry.keys())
            extra = set(entry.keys()) - REQUIRED_KEYS
            msg = f"[VALIDATE] ❌ Phần tử [{i}]: "
            if missing:
                msg += f"thiếu khóa {missing}. "
            if extra:
                msg += f"thừa khóa {extra}."
            print(msg)
            return False

        # --- Kiểm tra kiểu 'step' ---
        if not isinstance(entry["step"], int):
            print(f"[VALIDATE] ❌ Phần tử [{i}]: 'step' phải là int, nhận {type(entry['step'])}.")
            return False

        # --- Kiểm tra step liên tục ---
        expected_step = i + 1
        if entry["step"] != expected_step:
            print(f"[VALIDATE] ❌ Phần tử [{i}]: 'step' = {entry['step']}, kỳ vọng {expected_step}.")
            return False

        # --- Kiểm tra kiểu list cho 3 trường ---
        for key in ("unmasked_indices", "values", "margins"):
            if not isinstance(entry[key], list):
                print(f"[VALIDATE] ❌ Phần tử [{i}]: '{key}' phải là list, nhận {type(entry[key])}.")
                return False

        # --- Kiểm tra cùng độ dài ---
        n_ui = len(entry["unmasked_indices"])
        n_v = len(entry["values"])
        n_m = len(entry["margins"])
        if not (n_ui == n_v == n_m):
            print(
                f"[VALIDATE] ❌ Phần tử [{i}]: "
                f"len(unmasked_indices)={n_ui}, len(values)={n_v}, len(margins)={n_m} "
                f"không bằng nhau."
            )
            return False

        # --- Kiểm tra kiểu phần tử ---
        for idx_val in entry["unmasked_indices"]:
            if not isinstance(idx_val, (int, np.integer)):
                print(f"[VALIDATE] ❌ Phần tử [{i}]: unmasked_indices chứa giá trị không phải int.")
                return False

        for v in entry["values"]:
            if not isinstance(v, (int, np.integer)):
                print(f"[VALIDATE] ❌ Phần tử [{i}]: values chứa giá trị không phải int.")
                return False

        for m in entry["margins"]:
            if not isinstance(m, (float, int, np.floating)):
                print(f"[VALIDATE] ❌ Phần tử [{i}]: margins chứa giá trị không phải float.")
                return False

        # --- Kiểm tra board_state ---
        bs = entry["board_state"]
        if not isinstance(bs, np.ndarray):
            print(f"[VALIDATE] ❌ Phần tử [{i}]: 'board_state' phải là np.ndarray, nhận {type(bs)}.")
            return False
        if bs.shape != (seq_len,):
            print(f"[VALIDATE] ❌ Phần tử [{i}]: board_state.shape = {bs.shape}, kỳ vọng ({seq_len},).")
            return False

    print(f"[VALIDATE] ✅ step_history hợp lệ! ({len(step_history)} bước)")
    return True


# ---------------------------------------------------------------------------
# Self-test khi chạy trực tiếp
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("INTERFACES.PY – SELF TEST")
    print("=" * 60)

    # --- Test 1: DummyMDM ---
    print("\n[Test 1] DummyMDM")
    if _HAS_TORCH:
        model = DummyMDM(vocab_size=10)
        x = torch.randint(0, 10, (2, 81))  # batch=2, seq_len=81
        logits = model(x)
        assert logits.shape == (2, 81, 10), f"Shape sai: {logits.shape}"
        print(f"  ✅ logits.shape = {logits.shape}")
    else:
        print("  ⚠️ Bỏ qua test DummyMDM vì chưa cài đặt PyTorch (torch).")

    # --- Test 2: generate_mock_data ---
    print("\n[Test 2] generate_mock_data")
    quizzes, solutions = generate_mock_data(num_samples=5, n_size=9)
    assert quizzes.shape == (5, 81), f"quizzes shape sai: {quizzes.shape}"
    assert solutions.shape == (5, 81), f"solutions shape sai: {solutions.shape}"
    assert quizzes.dtype == np.int64, f"quizzes dtype sai: {quizzes.dtype}"
    assert solutions.dtype == np.int64, f"solutions dtype sai: {solutions.dtype}"
    assert np.all(solutions >= 1) and np.all(solutions <= 9), "solutions ngoài phạm vi [1,9]"
    assert np.any(quizzes == 0), "quizzes không có ô MASK nào"
    print(f"  ✅ quizzes.shape = {quizzes.shape}, solutions.shape = {solutions.shape}")
    print(f"  ✅ Số ô MASK trung bình: {np.mean(np.sum(quizzes == 0, axis=1)):.1f} / 81")

    # --- Test 3: generate_mock_step_history ---
    print("\n[Test 3] generate_mock_step_history")
    quiz = quizzes[0]
    step_history = generate_mock_step_history(quiz, n_size=9)
    num_masks = int(np.sum(quiz == 0))
    assert len(step_history) == num_masks, (
        f"Số bước ({len(step_history)}) != số MASK ({num_masks})"
    )
    print(f"  ✅ Số bước = {len(step_history)} (= số ô MASK)")
    print(f"  Bước đầu: {step_history[0]['step']}, "
          f"ô mở: {step_history[0]['unmasked_indices']}, "
          f"giá trị: {step_history[0]['values']}")

    # --- Test 4: validate_step_history ---
    print("\n[Test 4] validate_step_history")
    is_valid = validate_step_history(step_history, seq_len=81)
    assert is_valid, "step_history do generate_mock_step_history tạo ra phải hợp lệ!"
    print(f"  ✅ validate_step_history trả về True")

    # --- Test 5: validate với dữ liệu sai ---
    print("\n[Test 5] validate_step_history với dữ liệu sai")
    bad_history = [{"step": 1, "unmasked_indices": [0], "values": [1]}]  # thiếu khóa
    is_bad_valid = validate_step_history(bad_history, seq_len=81)
    assert not is_bad_valid, "Dữ liệu thiếu khóa phải trả về False!"
    print(f"  ✅ validate_step_history đúng phát hiện lỗi")

    # --- Test 6: DummyMDM với n_size nhỏ ---
    print("\n[Test 6] DummyMDM + mock data với n_size=4")
    q4, s4 = generate_mock_data(num_samples=3, n_size=4)
    assert q4.shape == (3, 16), f"Shape sai cho 4×4: {q4.shape}"
    if _HAS_TORCH:
        model4 = DummyMDM(vocab_size=5)
        logits4 = model4(torch.tensor(q4))
        assert logits4.shape == (3, 16, 5), f"Logits shape sai: {logits4.shape}"
    sh4 = generate_mock_step_history(q4[0], n_size=4)
    assert validate_step_history(sh4, seq_len=16)
    print(f"  ✅ Pipeline 4×4 hoạt động đúng")

    print("\n" + "=" * 60)
    print("TẤT CẢ TESTS ĐÃ PASS! ✅")
    print("=" * 60)
