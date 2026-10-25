"""Gộp kết quả chạy trên Colab (thư mục results tải từ Drive) vào results/ của laptop.

    python scripts/merge_colab.py D:/colab/rsf/results            # xem trước, chưa ghi gì
    python scripts/merge_colab.py D:/colab/rsf/results --apply    # gộp thật

- preds: chỉ chép file laptop chưa có. File trùng tên mà nội dung khác thì báo, không ghi đè.
- runs_*.csv: nối hai log, bỏ dòng trùng (model, series, target, split, seed, status).
- best_params.json: thêm model laptop chưa có. Model có ở cả hai mà tham số khác thì báo và giữ bản laptop.
- tuning/*.csv: chép file laptop chưa có.
- env_*.json: nối lịch sử, để paper ghi được model nào chạy trên máy nào.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from pathlib import Path

import pandas as pd

from rsf import results

KEY = ["model", "series", "target", "split", "seed", "status"]


def main(src: Path, apply: bool):
    dst = results.RESULTS
    if not (src / "preds").exists() and not (src / "best_params.json").exists():
        raise SystemExit(f"{src} không giống thư mục results (thiếu preds/ và best_params.json)")
    conflicts = []

    # preds
    new = []
    for f in sorted((src / "preds").rglob("*.parquet")) if (src / "preds").exists() else []:
        target = dst / f.relative_to(src)
        if not target.exists():
            new.append((f, target))
        elif not filecmp.cmp(f, target, shallow=False):
            conflicts.append(f"preds khác nội dung: {target.relative_to(dst)}")
    by_model = pd.Series([t.parent.name for _, t in new]).value_counts() if new else pd.Series(dtype=int)
    print(f"preds mới: {len(new)} file", {m: int(n) for m, n in by_model.items()})

    # tuning
    tun = []
    for f in sorted((src / "tuning").glob("*.csv")) if (src / "tuning").exists() else []:
        target = dst / "tuning" / f.name
        if not target.exists():
            tun.append((f, target))
        elif not filecmp.cmp(f, target, shallow=False):
            conflicts.append(f"tuning khác nội dung, giữ bản laptop: {f.name}")
    print("tuning mới:", [t.name for _, t in tun])

    # best_params
    bp_src = json.loads((src / "best_params.json").read_text()) if (src / "best_params.json").exists() else {}
    bp_dst = json.loads((dst / "best_params.json").read_text()) if (dst / "best_params.json").exists() else {}
    add = {m: p for m, p in bp_src.items() if m not in bp_dst}
    conflicts += [f"best_params khác, giữ bản laptop: {m}" for m, p in bp_src.items() if m in bp_dst and bp_dst[m] != p]
    print("best_params thêm:", add)

    # runs log
    logs = {}
    for f in src.glob("runs_*.csv"):
        old = pd.read_csv(dst / f.name) if (dst / f.name).exists() else pd.DataFrame()
        both = pd.concat([old, pd.read_csv(f)], ignore_index=True).drop_duplicates(subset=KEY, keep="last")
        logs[f.name] = both
        print(f"{f.name}: laptop {len(old)} dòng → sau khi gộp {len(both)} dòng")

    # env
    envs = {}
    for f in src.glob("env_*.json"):
        old = json.loads((dst / f.name).read_text()) if (dst / f.name).exists() else []
        seen = {e.get("time") for e in old}
        envs[f.name] = old + [e for e in json.loads(f.read_text()) if e.get("time") not in seen]

    for c in conflicts:
        print("CẢNH BÁO:", c)
    if not apply:
        print("\nChưa ghi gì. Kiểm tra số trên rồi chạy lại với --apply.")
        return
    for f, target in new + tun:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
    if add:
        (dst / "best_params.json").write_text(json.dumps({**bp_dst, **add}, indent=2))
    for name, t in logs.items():
        t.to_csv(dst / name, index=False)
    for name, e in envs.items():
        (dst / name).write_text(json.dumps(e, indent=2))
    print("Đã gộp vào", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="thư mục results tải từ Drive (MyDrive/rsf/results)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    main(a.src, a.apply)
