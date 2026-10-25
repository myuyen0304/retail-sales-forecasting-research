"""Chỉ số sai số và kiểm định so sánh model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def mape(y, p) -> float:
    """Phần trăm. Mọi chuỗi đã kiểm y > 0 (data.check) nên không cần epsilon."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if (y <= 0).any():
        raise ValueError("MAPE không xác định khi y ≤ 0")
    return float(np.mean(np.abs((y - p) / y)) * 100)


def scores(y, p) -> dict:
    return {"rmse": rmse(y, p), "mae": mae(y, p), "mape": mape(y, p)}


def diebold_mariano(e1, e2, h: int = 1, power: int = 1) -> tuple[float, float]:
    """DM với hiệu chỉnh mẫu nhỏ Harvey–Leybourne–Newbold. Thống kê âm: model 1 sai ít hơn.

    Trả (thống kê, p-value hai phía, phân phối t với n−1 bậc tự do).
    """
    d = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    return stat, float(2 * stats.t.sf(abs(stat), df=n - 1))


def friedman_nemenyi(table: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.Series, float, float]:
    """`table`: hàng = bài toán (chuỗi × fold), cột = model, giá trị = sai số (nhỏ là tốt).

    Trả (thứ hạng trung bình, p-value Friedman, critical difference của Nemenyi).
    """
    ranks = table.rank(axis=1, method="average")
    n, k = table.shape
    p = float(stats.friedmanchisquare(*[table[c].values for c in table]).pvalue)
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * n)))
    return ranks.mean().sort_values(), p, cd
