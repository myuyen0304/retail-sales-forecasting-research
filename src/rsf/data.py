"""Dựng chuỗi ngày TOTAL + 3 vùng từ CSV Datathon, và đọc lại cho thực nghiệm.

Revenue(d) = Σ quantity × unit_price (không trừ discount, gồm mọi trạng thái đơn);
COGS(d) = Σ quantity × products.cogs. Tổng 3 vùng phải khớp sales.csv từng ngày.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("RSF_RAW_DIR", ROOT / "data" / "raw"))
PROCESSED = Path(os.environ.get("RSF_PROCESSED_DIR", ROOT / "data" / "processed"))
RESULTS = Path(os.environ.get("RSF_RESULTS_DIR", ROOT / "results"))
SERIES_FILE = "daily_series.csv"
PROMO_FILE = "promotions.csv"
SERIES = ["TOTAL", "EAST", "CENTRAL", "WEST"]
REGIONS = SERIES[1:]
TARGETS = ["revenue", "cogs"]


def build(raw: Path = RAW, out: Path = PROCESSED, tol: float = 0.01) -> pd.DataFrame:
    items = pd.read_csv(raw / "order_items.csv", usecols=["order_id", "product_id", "quantity", "unit_price"])
    orders = pd.read_csv(raw / "orders.csv", usecols=["order_id", "order_date", "zip"])
    geo = pd.read_csv(raw / "geography.csv", usecols=["zip", "region"])
    products = pd.read_csv(raw / "products.csv", usecols=["product_id", "cogs"])
    x = (items.merge(orders, on="order_id", validate="m:1")
              .merge(geo, on="zip", how="left", validate="m:1")
              .merge(products, on="product_id", validate="m:1"))
    if x["region"].isna().any():
        raise ValueError("Có đơn không map được vùng")
    x["revenue"] = x["quantity"] * x["unit_price"]
    x["cogs"] = x["quantity"] * x["cogs"]
    x["series_id"] = x["region"].str.upper()
    reg = x.groupby(["order_date", "series_id"])[TARGETS].sum().reset_index()
    tot = reg.groupby("order_date")[TARGETS].sum().reset_index().assign(series_id="TOTAL")
    df = pd.concat([tot, reg]).rename(columns={"order_date": "date"})
    df["date"] = pd.to_datetime(df["date"])

    sales = pd.read_csv(raw / "sales.csv", parse_dates=["Date"]).set_index("Date")
    t = df[df.series_id == "TOTAL"].set_index("date")
    diff = max((t["revenue"] - sales["Revenue"]).abs().max(), (t["cogs"] - sales["COGS"]).abs().max())
    if not t.index.equals(sales.index) or diff > tol:
        raise ValueError(f"TOTAL không khớp sales.csv (lệch tối đa {diff})")

    df = df.sort_values(["series_id", "date"]).reset_index(drop=True)
    check(df)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / SERIES_FILE, index=False)
    pd.read_csv(raw / PROMO_FILE).to_csv(out / PROMO_FILE, index=False)
    return df


def check(df: pd.DataFrame) -> None:
    """Đủ ngày liên tục, giá trị dương (cần cho log và MAPE), vùng cộng lại bằng TOTAL."""
    for sid, g in df.groupby("series_id"):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        if len(g) != len(full) or not (g[TARGETS] > 0).all().all():
            raise ValueError(f"Chuỗi {sid} thiếu ngày hoặc có giá trị ≤ 0")
    wide = df.pivot(index="date", columns="series_id", values=TARGETS)
    for t in TARGETS:
        gap = (wide[t][REGIONS].sum(axis=1) - wide[t]["TOTAL"]).abs().max()
        if gap > 0.01:
            raise ValueError(f"3 vùng không cộng về TOTAL ({t}, lệch {gap})")


def load(processed: Path = PROCESSED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả (series dạng dài, promotions). Đây là hai file duy nhất cần upload lên Colab."""
    df = pd.read_csv(processed / SERIES_FILE, parse_dates=["date"])
    check(df)
    promos = pd.read_csv(processed / PROMO_FILE, parse_dates=["start_date", "end_date"])
    return df, promos


def series(df: pd.DataFrame, sid: str, target: str) -> pd.Series:
    s = df[df.series_id == sid].set_index("date")[target]
    s.index.freq = "D"
    return s


if __name__ == "__main__":
    d = build()
    print(d.groupby("series_id").agg(days=("date", "size"), first=("date", "min"), last=("date", "max"),
                                     revenue_sum=("revenue", "sum")))
