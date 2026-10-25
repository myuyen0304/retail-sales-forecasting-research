"""Seasonal naive: chép giá trị cùng ngày của 1 hoặc 2 năm trước (2 năm giữ tính chẵn/lẻ của năm)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rsf.models.base import Forecaster


def _same_day(year: int, d: pd.Timestamp) -> pd.Timestamp:
    day = min(d.day, pd.Timestamp(year=year, month=d.month, day=1).days_in_month)
    return pd.Timestamp(year=year, month=d.month, day=day)


class SeasonalNaive(Forecaster):
    family = "baseline"
    years = 1

    def fit(self, y, cal):
        self.y = y
        return self

    def predict(self, dates):
        last = self.y.index.max()
        out = []
        for d in dates:
            k = self.years
            src = _same_day(d.year - k, d)
            while src > last:
                k += self.years
                src = _same_day(d.year - k, d)
            out.append(self.y.loc[src])
        return np.asarray(out, float)


class SNaive1Y(SeasonalNaive):
    name, years = "snaive1y", 1


class SNaive2Y(SeasonalNaive):
    name, years = "snaive2y", 2
