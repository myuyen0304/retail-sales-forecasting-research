"""Export actual bằng callback SELECT, chỉ thực thi trên cloud."""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re

from cloud_pilot import cloud_runtime, sha256, validate_data, write_json


def quoted(identifier):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Identifier không hợp lệ.")
    return chr(96) + identifier + chr(96)


def export_actual(query, catalog, release_id, output):
    cloud_runtime()
    if not re.fullmatch(r"[a-f0-9]{24}", release_id):
        raise ValueError("Cần release_id 24 ký tự hex đã chọn.")
    catalog_sql = quoted(catalog)
    state_sql = "SELECT payload FROM " + catalog_sql + ".retail_control.state WHERE name = 'main'"
    state_frame = query(state_sql)
    if len(state_frame) != 1:
        raise ValueError("Registry phải có đúng một bản ghi main.")
    payload = state_frame.iloc[0]["payload"]
    state = json.loads(payload) if isinstance(payload, str) else payload
    record = state.get("releases", {}).get(release_id)
    schema = "r_" + release_id + "_reporting"
    if not record or record.get("status") != "ready" or record.get("schema") != schema or record.get("release_id") != release_id:
        raise ValueError("Release không tồn tại hoặc chưa ready.")
    relation = catalog + "." + schema + ".rpt_forecast_actual"
    sql_relation = catalog_sql + "." + quoted(schema) + ".rpt_forecast_actual"
    select_sql = "SELECT sale_date, g, cogs FROM " + sql_relation + " ORDER BY sale_date"
    frame = query(select_sql)
    if len(frame) != 3833:
        raise ValueError("Số dòng khác contract 3833; chưa tạo snapshot.")
    audit_sql = ("SELECT COUNT(*) AS row_count, MIN(sale_date) AS first_date, "
                 "MAX(sale_date) AS last_date, SUM(g) AS total_g, SUM(cogs) AS total_cogs FROM "
                 + sql_relation)
    audit = query(audit_sql)
    if len(audit) != 1 or int(audit.iloc[0]["row_count"]) != len(frame):
        raise ValueError("Đối soát số dòng thất bại.")
    for column in ("g", "cogs"):
        total = sum((Decimal(str(value)) for value in frame[column]), Decimal("0"))
        expected = Decimal(str(audit.iloc[0]["total_" + column]))
        if abs(total - expected) > Decimal("0.01"):
            raise ValueError("Đối soát tổng thất bại: " + column)
    monthly_sql = ("SELECT date_trunc('MONTH', sale_date) AS month, SUM(g) AS g, SUM(cogs) AS cogs FROM "
                   + sql_relation + " GROUP BY 1 ORDER BY 1")
    monthly = query(monthly_sql)
    data = validate_data(frame)
    import pandas as pd
    local_monthly = data.assign(month=data["sale_date"].dt.to_period("M")).groupby("month")[["g", "cogs"]].sum()
    monthly["month"] = pd.to_datetime(monthly["month"]).dt.to_period("M")
    remote_monthly = monthly.set_index("month")[["g", "cogs"]].astype(float)
    if not local_monthly.index.equals(remote_monthly.index):
        raise ValueError("Đối soát tập tháng thất bại.")
    if ((local_monthly - remote_monthly).abs() > 0.01).any().any():
        raise ValueError("Đối soát tổng tháng thất bại.")
    root = Path(output) / release_id
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "actual.csv"
    if csv_path.exists() or (root / "manifest.json").exists():
        raise FileExistsError("Snapshot đã có; dùng lại hoặc chọn thư mục mới, không ghi đè.")
    data.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    digest = sha256(csv_path)
    manifest = {
        "snapshot_id": digest[:20], "sha256": digest, "release_id": release_id,
        "source_relation": relation, "actual_only": True,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(data), "first_date": str(data.sale_date.min().date()),
        "last_date": str(data.sale_date.max().date()),
        "total_g": str(audit.iloc[0]["total_g"]), "total_cogs": str(audit.iloc[0]["total_cogs"]),
        "monthly_reconciled": True,
        "queries": [state_sql, select_sql, audit_sql, monthly_sql],
    }
    write_json(root / "manifest.json", manifest)
    return str(csv_path), str(root / "manifest.json")
