"""Pilot CPU chỉ chạy trên Colab/Databricks; không dùng code hệ thống chính."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import pandas as pd

DEVELOPMENT_END = pd.Timestamp("2021-07-01")


def cloud_runtime():
    # Guard chống chạy nhầm; đây không phải cơ chế chứng thực hạ tầng.
    if os.getenv("COLAB_RELEASE_TAG") and Path("/content").is_dir():
        return "colab"
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return "databricks"
    raise RuntimeError("Không train local. Chạy trên runtime Colab hoặc Databricks.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_data(frame):
    required = {"sale_date", "g", "cogs"}
    if not required.issubset(frame.columns):
        raise ValueError("CSV phải có sale_date,g,cogs từ rpt_forecast_actual.")
    data = frame[list(sorted(required))].copy()
    data["sale_date"] = pd.to_datetime(data["sale_date"], errors="raise")
    dates = data["sale_date"]
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("sale_date phải là ngày không có timezone/giờ.")
    data = data.sort_values("sale_date").reset_index(drop=True)
    expected = pd.date_range("2012-07-04", "2022-12-31", freq="D")
    if not pd.DatetimeIndex(data["sale_date"]).equals(expected):
        raise ValueError("Lịch actual không khớp 3833 ngày đã đăng ký; kiểm lại snapshot.")
    for name in ("g", "cogs"):
        data[name] = pd.to_numeric(data[name], errors="raise")
        if not np.isfinite(data[name]).all() or (data[name] < 0).any():
            raise ValueError("Target thiếu, âm hoặc không hữu hạn; cần điều tra dữ liệu.")
    return data


def load_snapshot(csv_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for key in ("snapshot_id", "release_id", "source_relation", "exported_at",
                "sha256", "actual_only"):
        if key not in manifest:
            raise ValueError("Manifest thiếu " + key)
    if manifest["actual_only"] is not True:
        raise ValueError("Manifest chưa xác nhận actual_only.")
    relation_parts = manifest["source_relation"].split(".")
    if (len(relation_parts) != 3 or relation_parts[-1] != "rpt_forecast_actual"
            or relation_parts[-2] != "r_" + manifest["release_id"] + "_reporting"):
        raise ValueError("Manifest phải chỉ đúng rpt_forecast_actual của release.")
    if sha256(csv_path) != manifest["sha256"]:
        raise ValueError("Checksum CSV khác manifest.")
    return validate_data(pd.read_csv(csv_path)), manifest


def score(y, prediction, train):
    y, prediction, train = [np.asarray(v, dtype=float) for v in (y, prediction, train)]
    if y.shape != prediction.shape or y.size == 0:
        raise ValueError("Prediction và actual phải cùng shape, không rỗng.")
    if not all(np.isfinite(v).all() for v in (y, prediction, train)):
        raise ValueError("Metric không nhận NaN/Inf.")
    error = prediction - y
    mae = float(np.abs(error).mean())
    rse_den = float(np.square(y - y.mean()).sum())
    mase_den = float(np.abs(train[365:] - train[:-365]).mean()) if len(train) > 365 else 0
    wape_den = float(np.abs(y).sum())
    nonzero = y != 0
    return {
        "n": len(y), "mae": mae, "rmse": float(np.sqrt(np.square(error).mean())),
        "rse": float(np.square(error).sum() / rse_den) if rse_den else None,
        "mape": float(100 * np.mean(np.abs(error / y))) if nonzero.all() else None,
        "mape_nonzero": float(100 * np.mean(np.abs(error[nonzero] / y[nonzero]))) if nonzero.any() else None,
        "mape_coverage": float(nonzero.mean()),
        "mase": mae / mase_den if mase_den else None,
        "wape": float(100 * np.abs(error).sum() / wape_den) if wape_den else None,
        "bias": float(error.mean()), "total_error": float(error.sum()),
    }


def calendar(dates):
    dates = pd.DatetimeIndex(dates)
    result = pd.DataFrame(index=dates)
    # Mã hóa cố định, không học từ validation.
    for month in range(2, 13):
        result["month_" + str(month)] = (dates.month == month).astype(float)
    result["day"] = dates.day / 31.0
    result["days_in_month"] = dates.days_in_month / 31.0
    result["days_to_end"] = (dates.days_in_month - dates.day) / 31.0
    phase = 2 * np.pi * (dates.dayofyear - 1) / np.where(dates.is_leap_year, 366, 365)
    for harmonic in (1, 2, 3):
        result["sin_" + str(harmonic)] = np.sin(harmonic * phase)
        result["cos_" + str(harmonic)] = np.cos(harmonic * phase)
    return result


def history_row(values, config):
    row = [values[-lag] for lag in config["lag_days"]]
    for window in config["rolling_days"]:
        row.extend([np.mean(values[-window:]), np.std(values[-window:], ddof=0)])
    return row


def supervised(train, config):
    first = max(config["lag_days"] + config["rolling_days"])
    values = train.to_numpy(dtype=float)
    if len(values) <= first:
        raise ValueError("Không đủ train cho lag đã đăng ký.")
    cal = calendar(train.index).to_numpy()
    rows = [history_row(values[:i], config) + list(cal[i]) for i in range(first, len(values))]
    return np.asarray(rows), values[first:], first


def baseline(train, future, model):
    values = train.to_numpy(dtype=float)
    if model == "mean":
        return np.repeat(values.mean(), len(future))
    if model == "naive":
        return np.repeat(values[-1], len(future))
    if model == "drift":
        return values[-1] + np.arange(1, len(future) + 1) * (values[-1] - values[0]) / (len(values) - 1)
    known = train.to_dict()
    predictions = []
    for date in future:
        if model == "snaive7":
            reference = date - pd.Timedelta(days=7)
        else:
            years = 1 if model == "snaive_year" else 2
            reference = date - pd.DateOffset(years=years)
        value = float(known[reference])
        predictions.append(value)
        known[date] = max(value, 0.0)
    return np.asarray(predictions)


def fit_predict(train, future, model_name, config):
    cloud_runtime()  # Kiểm trước mọi đường fit.
    start = time.perf_counter()
    fitted = None
    fitted_dates = None
    metadata = {"input_set": "F0", "strategy": "native", "point": "point"}
    estimator = None
    baseline_names = {"mean", "naive", "drift", "snaive7", "snaive_year", "snaive_two_year"}
    if model_name in baseline_names:
        fit_seconds = time.perf_counter() - start
        pred_start = time.perf_counter()
        prediction = baseline(train, future, model_name)
        estimator = {"method": model_name, "history": train.to_dict()}
    elif model_name in {"ridge", "lightgbm"}:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        features, labels, first = supervised(train, config)
        feature_seconds = time.perf_counter() - start
        start = time.perf_counter()
        if model_name == "ridge":
            estimator = make_pipeline(StandardScaler(), Ridge(**config["ridge"]))
        else:
            from lightgbm import LGBMRegressor
            estimator = LGBMRegressor(random_state=config["seed"], **config["lightgbm"])
        estimator.fit(features, labels)
        fit_seconds = time.perf_counter() - start
        fitted = estimator.predict(features)
        fitted_dates = train.index[first:]
        history = list(train.to_numpy(dtype=float))
        cal = calendar(future).to_numpy()
        pred_start = time.perf_counter()
        prediction = []
        for i in range(len(future)):
            row = np.asarray([history_row(history, config) + list(cal[i])])
            value = float(estimator.predict(row)[0])
            prediction.append(value)
            history.append(max(value, 0.0))
        prediction = np.asarray(prediction)
        metadata.update(input_set="F1", strategy="recursive", feature_seconds=feature_seconds)
    elif model_name == "log_sarimax":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        options = config["log_sarimax"]
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            estimator = SARIMAX(np.log1p(train.to_numpy()), exog=calendar(train.index).to_numpy(),
                                order=tuple(options["order"]), trend=options["trend"],
                                enforce_stationarity=True, enforce_invertibility=True).fit(
                                    disp=False, maxiter=options["maxiter"])
        metadata["warnings"] = [str(w.message) for w in observed]
        if not estimator.mle_retvals.get("converged", False):
            raise RuntimeError("SARIMAX không hội tụ; ghi failed, không nhận score.")
        fit_seconds = time.perf_counter() - start
        fitted = np.expm1(np.asarray(estimator.fittedvalues))
        fitted_dates = train.index
        pred_start = time.perf_counter()
        prediction = np.expm1(np.asarray(estimator.forecast(len(future), exog=calendar(future).to_numpy())))
        metadata.update(input_set="F1", point="inverse_log_location", strategy="native")
    else:
        raise ValueError("Model chưa triển khai: " + model_name)
    pred_seconds = time.perf_counter() - pred_start
    if not np.isfinite(prediction).all():
        raise ValueError("Prediction không hữu hạn.")
    return prediction, fitted, fitted_dates, estimator, {
        **metadata, "fit_seconds": fit_seconds, "predict_seconds": pred_seconds,
    }


def make_figure(train, actual, prediction, fitted, fitted_dates, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(train.index, train.values, label="Actual train", linewidth=0.8)
    if fitted is not None:
        axes[0].plot(fitted_dates, np.maximum(fitted, 0), label="Fitted (in-sample)", alpha=0.7, linewidth=0.7)
    axes[0].set_title("Train — chỉ dùng chẩn đoán")
    axes[1].plot(actual.index, actual.values, label="Actual validation")
    axes[1].plot(actual.index, prediction, label="Forecast từ origin cố định")
    axes[1].set_title(title + " — development F5, không phải final test")
    for axis in axes:
        axis.set_ylabel("Giá trị tiền theo nguồn")
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(csv_path, manifest_path, config_path, output):
    runtime = cloud_runtime()
    data, source = load_snapshot(csv_path, manifest_path)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    origin = pd.Timestamp(config["origin"])
    if origin != pd.Timestamp("2019-12-31") or config["horizons"] != [30, 548]:
        raise ValueError("Pilot chỉ chạy F5, H=30/548; không mở final test.")
    if config["targets"] != ["G", "COGS"] or len(set(config["models"])) != len(config["models"]):
        raise ValueError("Config target/model không hợp lệ.")
    if origin + pd.Timedelta(days=max(config["horizons"])) > DEVELOPMENT_END:
        raise ValueError("Validation vượt development.")
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "statsmodels", "matplotlib", "joblib"):
        packages[name] = importlib.metadata.version(name)
    identity = {
        "snapshot_sha256": source["sha256"], "config_sha256": sha256(config_path),
        "code_sha256": sha256(__file__), "runtime": runtime,
        "python": platform.python_version(), "packages": packages,
        "machine": platform.machine(), "cpu_count": os.cpu_count(),
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    root = Path(output) / run_id
    root.mkdir(parents=True, exist_ok=True)
    run_manifest = {"run_id": run_id, "source": source, "config": config, **identity}
    write_json(root / "manifest.json", run_manifest)
    actual_data = data.set_index("sale_date")
    import joblib
    results = []
    for target in config["targets"]:
        series = actual_data["g" if target == "G" else "cogs"]
        train = series.loc[:origin].copy()
        for horizon in config["horizons"]:
            dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
            actual = series.reindex(dates)
            for model in config["models"]:
                task = root / (target + "_" + str(horizon) + "_" + model)
                task.mkdir(exist_ok=True)
                status_path = task / "status.json"
                if status_path.exists():
                    old = json.loads(status_path.read_text(encoding="utf-8"))
                    hashes = old.get("artifact_sha256", {})
                    if old.get("status") == "success" and hashes and all(
                            (task / name).exists() and sha256(task / name) == digest
                            for name, digest in hashes.items()):
                        results.append(old)
                        continue
                record = {
                    "run_id": run_id, "snapshot_id": source["snapshot_id"], "model_id": model,
                    "target": target, "origin": str(origin.date()), "horizon": horizon,
                    "fold_id": "F5", "split": "development", "seed": config["seed"],
                }
                write_json(status_path, {**record, "status": "running"})
                try:
                    raw, fitted, fitted_dates, estimator, timing = fit_predict(train, dates, model, config)
                    prediction = np.maximum(raw, 0)
                    predictions = pd.DataFrame({
                        **{key: [value] * horizon for key, value in record.items()},
                        "ds": dates, "lead_time": np.arange(1, horizon + 1),
                        "y_true": actual.to_numpy(), "y_pred_raw": raw, "y_pred": prediction,
                    })
                    predictions.to_csv(task / "predictions.csv", index=False)
                    joblib.dump(estimator, task / "model.joblib")
                    if fitted is not None:
                        pd.DataFrame({"ds": fitted_dates, "fitted_raw": fitted}).to_csv(task / "fitted.csv", index=False)
                    make_figure(train, actual, prediction, fitted, fitted_dates, task / "plot.png",
                                target + " / " + model + " / H=" + str(horizon))
                    artifact_names = ["predictions.csv", "model.joblib", "plot.png"]
                    if fitted is not None:
                        artifact_names.append("fitted.csv")
                    record.update(status="success", **score(actual.to_numpy(), prediction, train.to_numpy()),
                                  **timing, clipped_negative=int((raw < 0).sum()),
                                  artifact_sha256={name: sha256(task / name) for name in artifact_names})
                except Exception as exc:
                    record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                write_json(status_path, record)
                results.append(record)
                print(target, horizon, model, record["status"], flush=True)
                pd.DataFrame(results).drop(columns=["artifact_sha256"], errors="ignore").to_csv(
                    root / "metrics.csv", index=False)
    pd.DataFrame(results).drop(columns=["artifact_sha256"], errors="ignore").to_csv(root / "metrics.csv", index=False)
    write_json(root / "summary.json", {"total": len(results),
                                     "success": sum(r["status"] == "success" for r in results),
                                     "failed": sum(r["status"] != "success" for r in results)})
    return str(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/pilot_cpu.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run(args.csv, args.manifest, args.config, args.output))
