"""Kết hợp dự báo cho hồi quy (bản tương đương "voting" của phân loại).

Ba cách, áp lên dự báo đã lưu của các model calendar-aware:
- `avg_topk`: trung bình đều của k model có MAE CV thấp nhất (Voting Regressor);
- `inv_mae`: trọng số tỷ lệ nghịch với MAE;
- `stack_nnls`: hồi quy không âm (NNLS) trên dự báo out-of-fold, trọng số chuẩn hóa về tổng 1.

Chống rò rỉ: ở fold Fk, xếp hạng/trọng số chỉ học từ các fold trước Fk (F1 chưa có fold trước → trung bình đều
mọi model ứng viên). Ở test, học từ cả 4 fold CV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from rsf import metrics

METHODS = ["avg_topk", "inv_mae", "stack_nnls"]


def weights(method: str, history: pd.DataFrame, candidates: list[str], k: int = 3) -> dict[str, float]:
    """`history`: dự báo dạng dài (model, date, y_true, y_pred) của các fold được phép dùng."""
    if history.empty:
        return {m: 1 / len(candidates) for m in candidates}
    mae = history.groupby("model").apply(lambda g: metrics.mae(g.y_true, g.y_pred)).reindex(candidates).dropna()
    if method == "avg_topk":
        top = mae.nsmallest(k).index
        return {m: 1 / len(top) for m in top}
    if method == "inv_mae":
        top = mae.nsmallest(k).index
        w = 1 / mae[top]
        return (w / w.sum()).to_dict()
    wide = history.pivot_table(index=["split", "date"], columns="model", values="y_pred")[list(mae.index)]
    truth = history.groupby(["split", "date"])["y_true"].first().reindex(wide.index)
    scale = truth.mean()
    w, _ = nnls(wide.values / scale, truth.values / scale)
    if w.sum() == 0:
        return {m: 1 / len(mae) for m in mae.index}
    return {m: float(v / w.sum()) for m, v in zip(mae.index, w) if v > 0}


def combine(preds: pd.DataFrame, w: dict[str, float]) -> pd.DataFrame:
    """`preds`: dự báo dạng dài của một (chuỗi, target, split). Trả date, y_true, y_pred của bản gộp."""
    wide = preds.pivot_table(index="date", columns="model", values="y_pred")
    y = preds.groupby("date")["y_true"].first()
    p = sum(wide[m] * v for m, v in w.items())
    return pd.DataFrame({"date": wide.index, "y_true": y.reindex(wide.index).values, "y_pred": p.values})
