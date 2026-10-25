"""Biến suy từ lịch, dùng chung cho mọi model (calendar-aware).

Chép ý từ Calendar của hệ thống Retail Analytics, có hai thay đổi chống rò rỉ:
- luật lặp của khuyến mãi chỉ học từ các đợt bắt đầu trước hoặc tại origin;
- cờ chế độ `post2019` chỉ có khi origin đã qua mốc đứt gãy (lúc đó mới quan sát được).
Bộ biến cố định trước, không chọn biến bằng phần dư trên toàn chuỗi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_FOURIER = 5
REGIME_START = pd.Timestamp("2019-01-01")
REGIME_MIN_DAYS = 180  # cần ít nhất nửa năm sau mốc thì cờ chế độ mới có dữ liệu để học


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


class Calendar:
    def __init__(self, promotions: pd.DataFrame, origin: pd.Timestamp):
        self.origin = pd.Timestamp(origin)
        p = promotions[promotions["start_date"] <= self.origin].copy()
        p["family"] = p["promo_name"].str.replace(r"\s+\d{4}$", "", regex=True)
        p["year"] = p["start_date"].dt.year
        mode_md = lambda s: s.dt.strftime("%m-%d").mode().iloc[0]
        rules = p.groupby("family").agg(start_md=("start_date", mode_md), end_md=("end_date", mode_md),
                                        first_year=("year", "min"), years=("year", lambda v: sorted(v)))
        # Chỉ coi là "năm lẻ" khi đã thấy ít nhất hai lần, đều năm lẻ
        rules["odd_only"] = rules["years"].apply(lambda v: len(v) >= 2 and all(y % 2 == 1 for y in v))
        rules["slug"] = [_slug(f) for f in rules.index]
        self.rules = rules
        self.use_regime = self.origin >= REGIME_START + pd.Timedelta(days=REGIME_MIN_DAYS)

    def _window(self, fam: str, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        r = self.rules.loc[fam]
        start, end = pd.Timestamp(f"{year}-{r.start_md}"), pd.Timestamp(f"{year}-{r.end_md}")
        if end < start:
            end = pd.Timestamp(f"{year + 1}-{r.end_md}")
        return start, end

    def promo_flags(self, d: pd.DatetimeIndex) -> pd.DataFrame:
        X = pd.DataFrame(index=d)
        for fam, r in self.rules.iterrows():
            flag, since = np.zeros(len(d)), np.full(len(d), -1.0)
            for y in range(d.year.min() - 1, d.year.max() + 1):
                if y < r.first_year or (r.odd_only and y % 2 == 0):
                    continue
                s, e = self._window(fam, y)
                m = (d >= s) & (d <= e)
                flag[m], since[m] = 1.0, (d[m] - s).days
            X[f"promo_{r.slug}"], X[f"promo_{r.slug}_since"] = flag, since
        return X

    def features(self, dates, kind: str = "tree", weekday: bool = False, drop: tuple = ()) -> pd.DataFrame:
        """kind='linear': biến cộng được cho SARIMAX/Prophet; kind='tree': thêm biến thứ tự cho cây/mạng.

        `drop`: bỏ nhóm biến cho ablation theo nhóm — 'regime', 'promo' (gồm is_odd_year), 'dom' (ngày trong tháng).
        """
        d = pd.DatetimeIndex(dates)
        X = pd.DataFrame(index=d)
        doy = d.day_of_year.values
        for k in range(1, K_FOURIER + 1):
            X[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            X[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
        dim, dom = d.days_in_month.values, d.day.values
        X["days_to_eom"] = dim - dom
        for k in (1, 2, 3):
            X[f"is_last{k}"] = (dim - dom <= k - 1).astype(float)
            X[f"is_first{k}"] = (dom <= k).astype(float)
        pf = self.promo_flags(d)
        X = X.join(pf[[c for c in pf if not c.endswith("_since")]])
        if self.use_regime:
            X["post2019"] = (d >= REGIME_START).astype(float)
        if kind == "tree":
            X["month"], X["dom"], X["doy"] = d.month, dom, doy
            X["is_odd_year"] = (d.year % 2).astype(float)
            X = X.join(pf[[c for c in pf if c.endswith("_since")]])
        if weekday:
            for k in range(6):
                X[f"dow{k}"] = (d.day_of_week == k).astype(float)
        X = X.drop(columns=[c for c in X if any(c in GROUPS[g](X) for g in drop)])
        return X.astype(float)


GROUPS = {
    "regime": lambda X: ["post2019"],
    "promo": lambda X: [c for c in X if c.startswith("promo_")] + ["is_odd_year"],
    "dom": lambda X: ["days_to_eom", "dom"] + [c for c in X if c.startswith(("is_last", "is_first"))],
}
