"""Giao diện chung: model nhận chuỗi train (đơn vị tiền, > 0) và Calendar của origin, trả dự báo đơn vị tiền."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rsf.features import Calendar


class Forecaster:
    name = "base"
    family = "base"           # baseline | statistical | ml | dl
    calendar_aware = False
    stochastic = False         # True: chạy nhiều seed

    def __init__(self, seed: int = 42, **params):
        self.seed = seed
        self.params = params

    def fit(self, y: pd.Series, cal: Calendar) -> "Forecaster":
        raise NotImplementedError

    def predict(self, dates: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def fitted(self) -> pd.Series | None:
        """Giá trị khớp trên train (chỉ để vẽ hình chẩn đoán)."""
        return None
