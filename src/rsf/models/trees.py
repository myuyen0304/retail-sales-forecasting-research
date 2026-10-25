"""Random Forest, Gradient Boosting, XGBoost, LightGBM trên log(y).

Bản chính (calendar-aware): hồi quy trực tiếp từ biến lịch, không lag, nên không dồn sai số theo horizon.
Bản `_lag` (pure, ablation): lag + trung bình trượt, dự báo đệ quy từng ngày (cách của pilot cũ).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rsf.models.base import Forecaster

LAGS = [1, 2, 3, 7, 14, 28, 30, 60, 90, 180, 365]
ROLLS = [7, 28, 90]

DEFAULTS = {
    "rf": dict(n_estimators=500, min_samples_leaf=3, max_features=0.5),
    "gb": dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20),
    "xgboost": dict(n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8),
    "lightgbm": dict(n_estimators=800, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
}


def make_regressor(kind: str, seed: int, params: dict):
    p = {**DEFAULTS[kind], **params}
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **p)
    if kind == "gb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(random_state=seed, **p)
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=seed, n_jobs=4, tree_method="hist", **p)
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=seed, n_jobs=4, deterministic=True, force_row_wise=True, verbosity=-1, **p)


class _TreeCal(Forecaster):
    family = "ml"
    calendar_aware = True
    kind = "lightgbm"
    drop_groups: tuple = ()

    def fit(self, y, cal):
        self.cal, self.index = cal, y.index
        X = cal.features(y.index, "tree", drop=self.drop_groups)
        self.columns = list(X.columns)
        self.model = make_regressor(self.kind, self.seed, self.params).fit(X, np.log(y.values))
        return self

    def predict(self, dates):
        X = self.cal.features(dates, "tree", drop=self.drop_groups)[self.columns]
        return np.exp(self.model.predict(X))

    def fitted(self):
        return pd.Series(self.predict(self.index), index=self.index)


def lag_row(hist: list[float]) -> list[float]:
    return [hist[-k] for k in LAGS] + [float(np.mean(hist[-w:])) for w in ROLLS]


class _TreeLag(Forecaster):
    """Chỉ dùng quá khứ của chuỗi (log), dự báo đệ quy: giá trị dự báo đưa ngược vào lag của bước sau."""
    family = "ml"
    kind = "lightgbm"

    def fit(self, y, cal):
        self.index = y.index
        ly = np.log(y.values)
        first = max(LAGS + ROLLS)
        X = np.array([lag_row(list(ly[:t])) for t in range(first, len(ly))])
        self.model = make_regressor(self.kind, self.seed, self.params).fit(X, ly[first:])
        self.hist = list(ly)
        self._fitted = pd.Series(np.exp(self.model.predict(X)), index=y.index[first:])
        return self

    def predict(self, dates):
        hist = list(self.hist)
        out = []
        for _ in dates:
            v = float(self.model.predict(np.array([lag_row(hist)]))[0])
            hist.append(v)
            out.append(v)
        return np.exp(np.asarray(out))

    def fitted(self):
        return self._fitted


def _make(kind, cal_aware):
    base = _TreeCal if cal_aware else _TreeLag
    name = kind if cal_aware else f"{kind}_lag"
    return type(name, (base,), {"name": name, "kind": kind})


RF, GB, XGB, LGBM = (_make(k, True) for k in ("rf", "gb", "xgboost", "lightgbm"))
RF_LAG, GB_LAG, XGB_LAG, LGBM_LAG = (_make(k, False) for k in ("rf", "gb", "xgboost", "lightgbm"))
