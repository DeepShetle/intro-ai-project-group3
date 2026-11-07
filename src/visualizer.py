"""
src/visualizer.py – TV5: Trực quan hóa quá trình giải Sudoku
=============================================================
Cung cấp:
  - render_board_html   : Xuất bảng Sudoku dạng HTML/CSS có tô màu trực quan.
  - render_step_history : Phát lại từng bước mở ô trên Colab/Jupyter/terminal.
"""

from __future__ import annotations

import time
import numpy as np
from typing import List, Dict

# ---------------------------------------------------------------------------
# Kích thước box con theo n_size
# ---------------------------------------------------------------------------
BOX_DIMS = {
    4: (2, 2),   # 4×4 : box 2×2
    6: (2, 3),   # 6×6 : box 2×3
    9: (3, 3),   # 9×9 : box 3×3
}


# ---------------------------------------------------------------------------
# 1. render_board_html
# ---------------------------------------------------------------------------

def render_board_html(
    quiz_1d: np.ndarray,
    step_item: dict,
    n_size: int = 9,
) -> str:
    """Xuất bảng Sudoku dạng lưới HTML/CSS có kẻ khung khối con và tô màu.

    Quy tắc tô màu
    ----------------
    * **Xanh đậm** ``#1a5276``: ô đề bài cho trước (``quiz_1d != 0``).
    * **Xanh lá**  ``#27ae60``: ô vừa mở, margin ≥ 0.5 (tự tin cao).
    * **Cam**       ``#e67e22``: ô vừa mở, margin < 0.5 (tự tin thấp).
    * **Xám nhạt** ``#d5dbdb``: ô đã được mở ở các bước trước.
    * **Trắng**     ``#fdfefe``: ô vẫn còn [MASK].

    Thanh trạng thái hiển thị: bước suy luận, số ô vừa mở, margin TB,
    và số ô còn trống.

    Parameters
    ----------
    quiz_1d : np.ndarray, shape (L,), dtype int64
        Bảng đề gốc (0 = [MASK]).
    step_item : dict
        Một phần tử trong ``step_history`` với 5 khóa chuẩn Data Contract.
    n_size : int
        Kích thước bảng (4, 6, hoặc 9).

    Returns
    -------
    html : str
        Chuỗi HTML hiển thị bảng Sudoku.
    """
    box_r, box_c = BOX_DIMS.get(n_size, (int(n_size ** 0.5), int(n_size ** 0.5)))

    board = step_item["board_state"]
    unmasked_set = set(step_item["unmasked_indices"])
    margins = step_item["margins"]
    step_num = step_item["step"]

    # Margin lookup cho ô vừa mở
    margin_map = {
        idx: m
        for idx, m in zip(step_item["unmasked_indices"], margins)
    }

    # Thống kê cho thanh trạng thái
    avg_margin = float(np.mean(margins)) if len(margins) > 0 else 0.0
    num_unmasked = len(step_item["unmasked_indices"])
    num_remaining = int(np.sum(board == 0))

    # Kích thước ô theo bảng
    cell_px = 48 if n_size <= 6 else 40
    font_px = 22 if n_size <= 6 else 17

    # ── CSS ──────────────────────────────────────────────────
    margin_color = "#27ae60" if avg_margin >= 0.5 else "#e67e22"

    css = f"""
<style>
.sdk-container {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    display: inline-block; margin: 10px;
}}
.sdk-status {{
    background: #2c3e50; color: white;
    padding: 10px 16px; border-radius: 8px 8px 0 0;
    font-size: 14px; display: flex;
    justify-content: space-between; gap: 20px;
}}
.sdk-status .lbl {{ color: #95a5a6; font-size: 11px; }}
.sdk-status .val {{ font-weight: bold; font-size: 16px; }}
.sdk-grid {{
    border-collapse: collapse;
    border: 3px solid #2c3e50;
}}
.sdk-cell {{
    width: {cell_px}px; height: {cell_px}px;
    text-align: center; vertical-align: middle;
    font-size: {font_px}px; font-weight: bold;
    border: 1px solid #bdc3c7;
}}
.c-given  {{ background: #1a5276; color: #ffffff; }}
.c-new-hi {{ background: #27ae60; color: #ffffff; }}
.c-new-lo {{ background: #e67e22; color: #ffffff; }}
.c-prev   {{ background: #d5dbdb; color: #2c3e50; }}
.c-mask   {{ background: #fdfefe; color: #dcdde1; }}
</style>
"""

    # ── Thanh trạng thái ─────────────────────────────────────
    status = f"""
<div class="sdk-container">
  <div class="sdk-status">
    <span><span class="lbl">Bước</span><br><span class="val">#{step_num}</span></span>
    <span><span class="lbl">Ô vừa mở</span><br><span class="val">{num_unmasked}</span></span>
    <span><span class="lbl">Margin TB</span><br>
          <span class="val" style="color:{margin_color}">{avg_margin:.3f}</span></span>
    <span><span class="lbl">Còn trống</span><br><span class="val">{num_remaining}</span></span>
  </div>
  <table class="sdk-grid">
"""

    # ── Các hàng ─────────────────────────────────────────────
    rows: list[str] = []
    for r in range(n_size):
        cells: list[str] = []
        for c in range(n_size):
            idx = r * n_size + c
            val = int(board[idx])

            # Xác định class + nội dung hiển thị
            if quiz_1d[idx] != 0:
                cls = "c-given"
                txt = str(int(quiz_1d[idx]))
            elif idx in unmasked_set:
                cls = "c-new-hi" if margin_map.get(idx, 0.0) >= 0.5 else "c-new-lo"
                txt = str(val) if val != 0 else "?"
            elif val != 0:
                cls = "c-prev"
                txt = str(val)
            else:
                cls = "c-mask"
                txt = "&middot;"

            # Border khối con
            bdr: list[str] = []
            if r % box_r == 0 and r != 0:
                bdr.append("border-top:3px solid #2c3e50")
            if c % box_c == 0 and c != 0:
                bdr.append("border-left:3px solid #2c3e50")

            style = f' style="{";".join(bdr)}"' if bdr else ""
            cells.append(f'<td class="sdk-cell {cls}"{style}>{txt}</td>')

        rows.append("<tr>" + "".join(cells) + "</tr>")

    table_body = "\n".join(rows)
    closing = "</table></div>"

    return css + status + table_body + closing


# ---------------------------------------------------------------------------
# 2. render_step_history
# ---------------------------------------------------------------------------

def render_step_history(
    quiz_1d: np.ndarray,
    step_history: list,
    n_size: int = 9,
    delay: float = 0.25,
) -> None:
    """Phát lại trực quan từng bước mở ô trên Colab / Jupyter / terminal.

    Sử dụng ``IPython.display.clear_output(wait=True)`` để tạo hiệu ứng
    animation; nếu không có IPython, fallback in text trên terminal.

    Parameters
    ----------
    quiz_1d : np.ndarray, shape (L,), dtype int64
        Bảng đề gốc (0 = [MASK]).
    step_history : list
        Danh sách ``step_history`` theo Data Contract.
    n_size : int
        Kích thước bảng (4, 6, hoặc 9).
    delay : float
        Thời gian chờ giữa các bước (giây).
    """
    try:
        from IPython.display import display, HTML, clear_output
        _use_ipython = True
    except ImportError:
        _use_ipython = False

    total_masks = int(np.sum(quiz_1d == 0))
    total_steps = len(step_history)

    for i, step_item in enumerate(step_history):
        if _use_ipython:
            clear_output(wait=True)

            html = render_board_html(quiz_1d, step_item, n_size)

            pct = (i + 1) / total_steps * 100 if total_steps > 0 else 100
            filled = int(np.sum(step_item["board_state"] != 0))

            progress = f"""
<div style="font-family:sans-serif; margin:8px 10px; max-width:500px;">
  <div style="background:#ecf0f1; border-radius:4px; height:6px; overflow:hidden;">
    <div style="background:linear-gradient(90deg,#3498db,#2ecc71);
                height:6px; border-radius:4px; width:{pct:.1f}%;"></div>
  </div>
  <div style="color:#7f8c8d; font-size:11px; margin-top:4px; text-align:center;">
    Tiến trình: {i + 1}/{total_steps} bước &middot;
    {filled}/{n_size * n_size} ô đã điền
  </div>
</div>
"""
            display(HTML(html + progress))
        else:
            _render_terminal(quiz_1d, step_item, n_size)

        time.sleep(delay)

    # Thông báo hoàn tất
    if _use_ipython:
        print(f"\n🎉 Hoàn tất! Đã mở {total_masks} ô trong {total_steps} bước.")
    else:
        print(f"\nHoan tat! Da mo {total_masks} o trong {total_steps} buoc.")


# ---------------------------------------------------------------------------
# 3. _render_terminal (fallback khi không có IPython)
# ---------------------------------------------------------------------------

def _render_terminal(
    quiz_1d: np.ndarray,
    step_item: dict,
    n_size: int,
) -> None:
    """In bảng Sudoku ra terminal (fallback khi không có IPython)."""
    import os
    os.system("cls" if os.name == "nt" else "clear")

    box_r, box_c = BOX_DIMS.get(n_size, (int(n_size ** 0.5), int(n_size ** 0.5)))
    board = step_item["board_state"]
    unmasked_set = set(step_item["unmasked_indices"])
    step_num = step_item["step"]
    avg_margin = (
        float(np.mean(step_item["margins"]))
        if len(step_item["margins"]) > 0
        else 0.0
    )

    print(
        f"--- Buoc {step_num} | "
        f"O mo: {len(step_item['unmasked_indices'])} | "
        f"Margin TB: {avg_margin:.3f} ---"
    )

    seg_width = box_c * 4 - 1
    num_segs = n_size // box_c
    h_line = "+" + ("+".join(["-" * seg_width] * num_segs)) + "+"

    for r in range(n_size):
        if r % box_r == 0:
            print(h_line)
        row_str = "|"
        for c in range(n_size):
            idx = r * n_size + c
            val = int(board[idx])
            if c > 0 and c % box_c == 0:
                row_str += "|"
            if quiz_1d[idx] != 0:
                row_str += f" {val} "
            elif idx in unmasked_set:
                row_str += f"[{val}]"
            elif val != 0:
                row_str += f" {val} "
            else:
                row_str += " . "
        row_str += "|"
        print(row_str)
    print(h_line)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from interfaces import (
        generate_mock_data,
        generate_mock_step_history,
        validate_step_history,
    )

    print("=" * 50)
    print("VISUALIZER.PY – SELF TEST")
    print("=" * 50)

    for ns in (4, 9):
        print(f"\n[Test] Board {ns}x{ns}")
        quizzes, solutions = generate_mock_data(num_samples=1, n_size=ns)
        quiz = quizzes[0]
        history = generate_mock_step_history(quiz, n_size=ns)
        assert validate_step_history(history, seq_len=ns * ns)

        html = render_board_html(quiz, history[0], n_size=ns)
        assert "<table" in html, "HTML không chứa <table>"
        assert "sdk-status" in html, "HTML không chứa status bar"
        print(f"  ✅ render_board_html OK (len={len(html)})")

        # Terminal rendering (chỉ test bước cuối, không clear)
        _render_terminal(quiz, history[-1], n_size=ns)
        print(f"  ✅ _render_terminal OK")

    print("\n" + "=" * 50)
    print("TẤT CẢ TESTS ĐÃ PASS! ✅")
    print("=" * 50)
