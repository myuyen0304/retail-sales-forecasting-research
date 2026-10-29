"""Đóng gói paper để upload lên Overleaf (New Project → Upload Project).

    python scripts/make_overleaf_zip.py   → paper/overleaf.zip (không commit)

Zip gồm main.tex, refs.bib, tables/*.tex và các hình main.tex dùng (lấy từ results/figures vào figures/).
Chạy lại sau mỗi lần sửa paper hoặc dựng lại bảng/hình.
"""
from __future__ import annotations

import re
import zipfile

from rsf import data

PAPER = data.ROOT / "paper"
FIGURES = data.RESULTS / "figures"
OUT = PAPER / "overleaf.zip"


def main() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    missing = [f for f in figs if not (FIGURES / f).exists()]
    if missing:
        raise SystemExit(f"Thiếu hình trong results/figures: {missing}")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(PAPER / "main.tex", "main.tex")
        z.write(PAPER / "refs.bib", "refs.bib")
        for t in sorted((PAPER / "tables").glob("*.tex")):
            z.write(t, f"tables/{t.name}")
        for f in figs:
            z.write(FIGURES / f, f"figures/{f}")
    print(f"→ {OUT} ({len(figs)} hình)")


if __name__ == "__main__":
    main()
