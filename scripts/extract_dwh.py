"""Trích chuỗi TOTAL + 3 vùng và promotions từ DWH Databricks, rồi đối chiếu với bản dựng từ CSV.

    pip install -r requirements-dwh.txt
    python scripts/extract_dwh.py --profile retail-dev --release ccde5a5e21c54e9e9054066f

Ghi vào data/dwh/ (không ghi đè data/processed/): daily_series.csv, promotions.csv, manifest.json.
Revenue = Σ gross_amount (quantity × unit_price, không trừ discount, mọi trạng thái đơn); COGS = Σ cogs_amount.
Vùng lấy qua khách hàng (dim_customer → dim_geography); bản CSV lấy qua orders.zip, phần đối chiếu sẽ bắt nếu hai cách lệch.
Tài khoản chạy cần USE CATALOG, USE SCHEMA + SELECT trên schema marts và bảng reporting.rpt_forecast_actual của bản.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rsf import data

OUT = data.ROOT / "data" / "dwh"
TOL = 0.01  # cùng ngưỡng với data.build()

SERIES_SQL = """
select d.full_date as date, upper(g.region) as series_id,
       sum(f.gross_amount) as revenue, sum(f.cogs_amount) as cogs
from {s}_marts.fact_order_item f
join {s}_marts.dim_date d      on d.date_sk = f.date_sk
join {s}_marts.dim_customer c  on c.customer_sk = f.customer_sk
join {s}_marts.dim_geography g on g.geography_sk = c.geography_sk
group by d.full_date, upper(g.region)
"""
PROMO_SQL = """
select promo_id, promo_name, promo_type, cast(discount_value as double) as discount_value, start_date, end_date,
       applicable_category, promo_channel, cast(stackable_flag as int) as stackable_flag, min_order_value
from {s}_marts.dim_promotion order by promo_id
"""
ACTUAL_SQL = "select sale_date as date, g as revenue, cogs from {s}_reporting.rpt_forecast_actual"


def query(w, warehouse_id: str, sql: str) -> pd.DataFrame:
    from databricks.sdk.service.sql import Disposition, Format, StatementState
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=warehouse_id, wait_timeout="50s",
                                                disposition=Disposition.INLINE, format=Format.JSON_ARRAY)
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"Query lỗi ({r.status.state}): {r.status.error.message if r.status.error else ''}")
    if r.manifest.truncated:
        raise RuntimeError("Kết quả bị cắt (vượt giới hạn INLINE)")
    rows = list(r.result.data_array or [])
    for i in range(1, r.manifest.total_chunk_count or 1):
        rows += w.statement_execution.get_statement_result_chunk_n(r.statement_id, i).data_array or []
    return pd.DataFrame(rows, columns=[c.name for c in r.manifest.schema.columns])


def max_gap(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> tuple[int, float]:
    """(số dòng không ghép được, lệch tuyệt đối lớn nhất của revenue/cogs)."""
    m = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    unmatched = int((m["_merge"] != "both").sum())
    both = m[m["_merge"] == "both"]
    gap = max((both[f"{t}_a"] - both[f"{t}_b"]).abs().max() for t in data.TARGETS)
    return unmatched, float(gap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, help="profile Databricks CLI (~/.databrickscfg)")
    ap.add_argument("--release", required=True, help="mã bản dữ liệu, vd ccde5a5e21c54e9e9054066f")
    ap.add_argument("--catalog", default="retail_lab")
    ap.add_argument("--warehouse-id", help="mặc định: SQL warehouse đầu tiên của workspace")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient(profile=a.profile)
    wid = a.warehouse_id or next(iter(w.warehouses.list())).id
    s = f"{a.catalog}.r_{a.release}"

    df = query(w, wid, SERIES_SQL.format(s=s))
    df["date"] = pd.to_datetime(df["date"])
    df[data.TARGETS] = df[data.TARGETS].astype(float)
    tot = df.groupby("date")[data.TARGETS].sum().reset_index().assign(series_id="TOTAL")
    df = pd.concat([tot, df]).sort_values(["series_id", "date"]).reset_index(drop=True)[
        ["date", "revenue", "cogs", "series_id"]]
    data.check(df)

    promos = query(w, wid, PROMO_SQL.format(s=s))
    actual = query(w, wid, ACTUAL_SQL.format(s=s))
    actual["date"] = pd.to_datetime(actual["date"])
    actual[data.TARGETS] = actual[data.TARGETS].astype(float)

    # Đối chiếu: TOTAL với rpt_forecast_actual của cùng bản; cả 4 chuỗi và promotions với bản CSV hiện có
    report = {}
    n, gap = max_gap(df[df.series_id == "TOTAL"].drop(columns="series_id"), actual, ["date"])
    report["total_vs_rpt_forecast_actual"] = {"unmatched_rows": n, "max_abs_gap": gap}
    csv_series = data.PROCESSED / data.SERIES_FILE
    if csv_series.exists():
        ref = pd.read_csv(csv_series, parse_dates=["date"])
        for sid in data.SERIES:
            n, gap = max_gap(df[df.series_id == sid], ref[ref.series_id == sid], ["date", "series_id"])
            report[f"{sid}_vs_csv"] = {"unmatched_rows": n, "max_abs_gap": gap}
        report["promotions_vs_csv"] = None  # so sau khi ghi file: đọc lại cả hai như data.load()
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / data.SERIES_FILE, index=False)
    promos.to_csv(a.out / data.PROMO_FILE, index=False)
    if "promotions_vs_csv" in report:
        ref_p, got_p = (pd.read_csv(d / data.PROMO_FILE, parse_dates=["start_date", "end_date"])
                        for d in (data.PROCESSED, a.out))
        try:
            pd.testing.assert_frame_equal(ref_p, got_p, check_dtype=False)
            report["promotions_vs_csv"] = {"identical": True, "rows": len(got_p)}
        except AssertionError as e:
            report["promotions_vs_csv"] = {"identical": False, "rows": len(got_p), "diff": str(e)[:300]}
    ok = all(v.get("identical", v.get("unmatched_rows") == 0 and v.get("max_abs_gap", 0) <= TOL)
             for v in report.values())
    manifest = {"extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "workspace": w.config.host, "catalog": a.catalog, "release": a.release, "warehouse_id": wid,
                "user": w.current_user.me().user_name, "rows": len(df), "promotions": len(promos),
                "sql": {"series": SERIES_SQL.format(s=s).strip(), "promotions": PROMO_SQL.format(s=s).strip(),
                        "actual": ACTUAL_SQL.format(s=s)},
                "tolerance": TOL, "checks": report, "matches_csv": ok}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    for k, v in report.items():
        print(f"{k:32s} {v}")
    print(f"→ {a.out}  ({len(df)} dòng, {len(promos)} promotions). Khớp bản CSV: {'CÓ' if ok else 'KHÔNG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
