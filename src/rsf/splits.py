"""Phân chia thời gian: 4 fold CV cửa sổ mở rộng (validation 365 ngày) và một test chấm một lần."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

START = pd.Timestamp("2012-07-04")
TEST_ORIGIN = pd.Timestamp("2021-06-30")
TEST_END = pd.Timestamp("2022-12-31")
VAL_DAYS = 365


@dataclass(frozen=True)
class Split:
    name: str
    origin: pd.Timestamp     # ngày cuối cùng được thấy khi fit
    end: pd.Timestamp        # ngày cuối vùng chấm
    note: str = ""

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.origin + pd.Timedelta(days=1), self.end, freq="D")


def _fold(i: int, origin: str, note: str) -> Split:
    o = pd.Timestamp(origin)
    return Split(f"F{i}", o, o + pd.Timedelta(days=VAL_DAYS), note)


FOLDS = [
    _fold(1, "2017-06-30", "tháng 8/2017 năm lẻ, trước đứt gãy"),
    _fold(2, "2018-06-30", "chứa đứt gãy cuối 2018"),
    _fold(3, "2019-06-30", "tháng 8/2019 năm lẻ"),
    _fold(4, "2020-06-30", "= validation của sơ đồ train/val/test"),
]
TEST = Split("TEST", TEST_ORIGIN, TEST_END, "tháng 8/2021 năm lẻ; chấm một lần")
SPLITS = {s.name: s for s in FOLDS + [TEST]}

# Sơ đồ train/validation/test cho hình của paper (mục 3.4)
TRAIN_END = FOLDS[-1].origin
VAL_END = FOLDS[-1].end

LEAD_BUCKETS = [(1, 30), (31, 90), (91, 365), (366, 549)]


def check() -> None:
    for f in FOLDS:
        if f.end > TEST_ORIGIN:
            raise ValueError(f"{f.name} chồng vào test")
    if VAL_END != TEST_ORIGIN:
        raise ValueError("Validation phải kết thúc đúng ngày trước test")


check()
