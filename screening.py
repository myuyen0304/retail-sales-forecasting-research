"""Screening riêng cho các họ model người dùng yêu cầu; chỉ fit trên cloud."""
from __future__ import annotations

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

from cloud_pilot import (
    DEVELOPMENT_END, calendar, cloud_runtime, history_row, load_snapshot,
    score, sha256, supervised, write_json,
)

FOLDS = {
    "F1": "2017-12-31",
    "F2": "2018-06-30",
    "F3": "2018-12-31",
    "F4": "2019-06-30",
    "F5": "2019-12-31",
}
TABULAR = {"random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"}
STATISTICAL = {"arima", "arimax", "sarima", "sarimax"}
NEURAL = {"rnn", "lstm"}
SUPPORTED = TABULAR | STATISTICAL | NEURAL | {"snaive_year", "snaive_two_year"}


def validate_config(config):
    models = config["models"]
    folds = config["folds"]
    horizons = config["horizons"]
    if not models or len(models) != len(set(models)) or set(models) - SUPPORTED:
        raise ValueError("Danh sách model trống, trùng hoặc chưa triển khai.")
    if not folds or len(folds) != len(set(folds)) or set(folds) - FOLDS.keys():
        raise ValueError("Fold không hợp lệ.")
    if config["targets"] != ["G", "COGS"]:
        raise ValueError("Target phải là G và COGS.")
    if not horizons or len(horizons) != len(set(horizons)) or any(h not in (7, 30, 90, 365, 548) for h in horizons):
        raise ValueError("Horizon không hợp lệ.")
    if set(models) & NEURAL and config.get("runtime") != "gpu":
        raise ValueError("RNN/LSTM chỉ chạy trong cấu hình GPU.")
    if set(models) & (TABULAR | STATISTICAL) and config.get("runtime") != "cpu":
        raise ValueError("CPU model phải dùng cấu hình CPU.")
    for fold in folds:
        if pd.Timestamp(FOLDS[fold]) + pd.Timedelta(days=max(horizons)) > DEVELOPMENT_END:
            raise ValueError("Validation vượt development.")
    if config.get("seed") is None:
        raise ValueError("Thiếu seed.")


def fit_tabular(train, future, model_name, config):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    features, labels, first = supervised(train, config)
    start = time.perf_counter()
    options = config[model_name].copy()
    seed = int(config["seed"])
    if model_name == "random_forest":
        estimator = RandomForestRegressor(random_state=seed, **options)
    elif model_name == "extra_trees":
        estimator = ExtraTreesRegressor(random_state=seed, **options)
    elif model_name == "xgboost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(random_state=seed, **options)
    elif model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(random_state=seed, **options)
    else:
        from catboost import CatBoostRegressor
        estimator = CatBoostRegressor(random_seed=seed, **options)
    estimator.fit(features, labels)
    fit_seconds = time.perf_counter() - start
    fitted = estimator.predict(features)
    history = list(train.to_numpy(dtype=float))
    cal = calendar(future).to_numpy()
    start = time.perf_counter()
    prediction = []
    for row in cal:
        x = np.asarray([history_row(history, config) + list(row)])
        value = float(estimator.predict(x)[0])
        prediction.append(value)
        history.append(max(0.0, value))
    return np.asarray(prediction), fitted, train.index[first:], estimator, {
        "input_set": "F1", "strategy": "recursive",
        "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - start,
    }



def statistical_exog(dates):
    columns = [f"{kind}_{harmonic}" for harmonic in (1, 2, 3)
               for kind in ("sin", "cos")]
    return calendar(dates)[columns].to_numpy()


def fit_statistical(train, future, model_name, config):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    options = config["statistical"][model_name]
    use_exog = model_name in {"arimax", "sarimax"}
    seasonal = model_name in {"sarima", "sarimax"}
    exog_train = statistical_exog(train.index) if use_exog else None
    exog_future = statistical_exog(future) if use_exog else None
    seasonal_order = tuple(options["seasonal_order"]) if seasonal else (0, 0, 0, 0)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        estimator = SARIMAX(
            np.log1p(train.to_numpy(dtype=float)),
            exog=exog_train,
            order=tuple(options["order"]),
            seasonal_order=seasonal_order,
            trend=options.get("trend", "n"),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=options["maxiter"])
    if not estimator.mle_retvals.get("converged", False):
        raise RuntimeError(model_name + " không hội tụ sau " + str(options["maxiter"]) + " vòng.")
    fit_seconds = time.perf_counter() - start
    fitted = np.expm1(np.asarray(estimator.fittedvalues))
    start = time.perf_counter()
    prediction = np.expm1(np.asarray(estimator.forecast(len(future), exog=exog_future)))
    return prediction, fitted, train.index, estimator, {
        "input_set": "F1" if use_exog else "F0",
        "strategy": "native", "transform": "log1p_inverse_location",
        "seasonal_order": list(seasonal_order),
        "exog": "annual_fourier_3" if use_exog else None,
        "warning_count": len(observed),
        "fit_seconds": fit_seconds, "predict_seconds": time.perf_counter() - start,
    }


def fit_neural(train, future, model_name, config):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    if not torch.cuda.is_available():
        raise RuntimeError("Cấu hình GPU yêu cầu Colab GPU runtime có CUDA.")
    options = config["neural"]
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    lookback = int(options["lookback"])
    values = np.log1p(train.to_numpy(dtype=np.float64))
    center = float(values.mean())
    scale = float(values.std())
    if scale == 0:
        scale = 1.0
    normalized = ((values - center) / scale).astype(np.float32)
    if len(normalized) <= lookback + 20:
        raise ValueError("Không đủ train cho neural lookback.")
    x = np.stack([normalized[i-lookback:i] for i in range(lookback, len(normalized))])[:, :, None]
    y = normalized[lookback:, None]
    # Validation nội bộ là đoạn cuối train của fold, không dùng ngày sau origin.
    cut = max(1, int(len(x) * 0.8))
    if cut >= len(x):
        raise ValueError("Không đủ cửa sổ validation nội bộ.")
    device = torch.device("cuda")

    class SequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            cell = nn.RNN if model_name == "rnn" else nn.LSTM
            self.recurrent = cell(1, int(options["hidden_size"]), batch_first=True)
            self.head = nn.Linear(int(options["hidden_size"]), 1)

        def forward(self, batch):
            output, _ = self.recurrent(batch)
            return self.head(output[:, -1, :])

    estimator = SequenceModel().to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=float(options["learning_rate"]))
    criterion = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[:cut]), torch.from_numpy(y[:cut])),
        batch_size=int(options["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val = torch.from_numpy(x[cut:]).to(device)
    y_val = torch.from_numpy(y[cut:]).to(device)
    best = float("inf")
    patience = 0
    best_state = None
    epochs_run = 0
    start = time.perf_counter()
    for epoch in range(int(options["max_epochs"])):
        estimator.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(estimator(bx), by)
            loss.backward()
            optimizer.step()
        estimator.eval()
        with torch.inference_mode():
            loss_val = float(criterion(estimator(x_val), y_val).item())
        epochs_run = epoch + 1
        if loss_val < best - float(options["min_delta"]):
            best = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in estimator.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(options["patience"]):
                break
    estimator.load_state_dict(best_state)
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - start
    # Đường fitted trên train chỉ dùng chẩn đoán, không là điểm validation.
    estimator.eval()
    with torch.inference_mode():
        fitted_scaled = []
        for offset in range(0, len(x), 512):
            fitted_scaled.extend(estimator(torch.from_numpy(x[offset:offset+512]).to(device)).cpu().numpy().ravel())
    fitted = np.expm1(np.asarray(fitted_scaled) * scale + center)
    history = list(normalized)
    prediction = []
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in future:
            window = torch.tensor(history[-lookback:], dtype=torch.float32, device=device).view(1, lookback, 1)
            value_scaled = float(estimator(window).item())
            history.append(value_scaled)
            prediction.append(np.expm1(value_scaled * scale + center))
    torch.cuda.synchronize()
    artifact = {
        "state_dict": best_state, "model_name": model_name, "lookback": lookback,
        "hidden_size": int(options["hidden_size"]), "center": center, "scale": scale,
    }
    return np.asarray(prediction), fitted, train.index[lookback:], artifact, {
        "input_set": "F0", "strategy": "recursive", "epochs": epochs_run,
        "internal_val_loss": best, "fit_seconds": fit_seconds,
        "predict_seconds": time.perf_counter() - start,
    }


def fit_model(train, future, model_name, config):
    cloud_runtime()
    if model_name in TABULAR:
        return fit_tabular(train, future, model_name, config)
    if model_name in STATISTICAL:
        return fit_statistical(train, future, model_name, config)
    if model_name in NEURAL:
        return fit_neural(train, future, model_name, config)
    if model_name in {"snaive_year", "snaive_two_year"}:
        from cloud_pilot import baseline
        start = time.perf_counter()
        prediction = baseline(train, future, model_name)
        return prediction, None, None, {"method": model_name}, {
            "input_set": "F0", "strategy": "recursive", "fit_seconds": 0.0,
            "predict_seconds": time.perf_counter() - start,
        }
    raise ValueError("Model chưa triển khai: " + model_name)



def make_screening_figure(train, actual, prediction, fitted, fitted_dates, path, title, fold):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(train.index, train.values, label="Actual train", linewidth=0.8)
    if fitted is not None:
        axes[0].plot(fitted_dates, np.maximum(fitted, 0), label="Fitted train", alpha=0.7, linewidth=0.7)
    axes[0].set_title("Train - fitted chỉ dùng chẩn đoán")
    axes[1].plot(actual.index, actual.values, label="Actual validation")
    axes[1].plot(actual.index, prediction, label="Forecast từ origin cố định")
    axes[1].set_title(title + " - development " + fold)
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
    validate_config(config)
    if config["runtime"] == "gpu":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Hãy chọn Colab GPU runtime trước khi chạy RNN/LSTM.")
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "statsmodels", "matplotlib", "joblib"]:
        packages[name] = importlib.metadata.version(name)
    optional_packages = {"random_forest": [], "extra_trees": [], "xgboost": ["xgboost"],
                         "lightgbm": ["lightgbm"], "catboost": ["catboost"],
                         "rnn": ["torch"], "lstm": ["torch"]}
    for model in config["models"]:
        for name in optional_packages.get(model, []):
            packages[name] = importlib.metadata.version(name)
    identity = {
        "snapshot_sha256": source["sha256"], "config_sha256": sha256(config_path),
        "code_sha256": sha256(__file__), "base_code_sha256": sha256(Path(__file__).with_name("cloud_pilot.py")),
        "runtime": runtime, "python": platform.python_version(),
        "packages": packages, "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "gpu_name": (torch.cuda.get_device_name(0) if config["runtime"] == "gpu" else None),
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    root = Path(output) / run_id
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "manifest.json", {"run_id": run_id, "source": source, "config": config, **identity})
    series_frame = data.set_index("sale_date")
    results = []
    for fold in config["folds"]:
        origin = pd.Timestamp(FOLDS[fold])
        for target in config["targets"]:
            series = series_frame["g" if target == "G" else "cogs"]
            train = series.loc[:origin].copy()
            for horizon in config["horizons"]:
                future = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon)
                actual = series.reindex(future)
                if actual.isna().any():
                    raise ValueError("Thiếu actual validation.")
                for model in config["models"]:
                    task = root / f"{fold}_{target}_{horizon}_{model}"
                    task.mkdir(exist_ok=True)
                    status_path = task / "status.json"
                    if status_path.exists():
                        old = json.loads(status_path.read_text(encoding="utf-8"))
                        hashes = old.get("artifact_sha256", {})
                        if old.get("status") == "success" and hashes and all(
                            (task / name).exists() and sha256(task / name) == digest
                            for name, digest in hashes.items()
                        ):
                            results.append(old)
                            continue
                    record = {
                        "run_id": run_id, "snapshot_id": source["snapshot_id"],
                        "model_id": model, "target": target, "fold_id": fold,
                        "origin": str(origin.date()), "horizon": horizon,
                        "split": "development", "seed": config["seed"],
                    }
                    write_json(status_path, {**record, "status": "running"})
                    try:
                        raw, fitted, fitted_dates, estimator, timing = fit_model(train, future, model, config)
                        if raw.shape != (horizon,) or not np.isfinite(raw).all():
                            raise ValueError("Forecast sai độ dài hoặc có NaN/Inf.")
                        prediction = np.maximum(raw, 0)
                        pd.DataFrame({
                            **{k: [v] * horizon for k, v in record.items()},
                            "ds": future, "lead_time": np.arange(1, horizon+1),
                            "y_true": actual.to_numpy(), "y_pred_raw": raw,
                            "y_pred": prediction,
                        }).to_csv(task / "predictions.csv", index=False)
                        if model in NEURAL:
                            import torch
                            torch.save(estimator, task / "model.pt")
                            model_file = "model.pt"
                        else:
                            import joblib
                            joblib.dump(estimator, task / "model.joblib")
                            model_file = "model.joblib"
                        artifacts = ["predictions.csv", model_file]
                        if fitted is not None:
                            pd.DataFrame({"ds": fitted_dates, "fitted_raw": fitted}).to_csv(
                                task / "fitted.csv", index=False)
                            artifacts.append("fitted.csv")
                        make_screening_figure(train, actual, prediction, fitted, fitted_dates,
                                              task / "plot.png", f"{target} / {model} / H={horizon}", fold)
                        artifacts.append("plot.png")
                        record.update(
                            status="success", **score(actual.to_numpy(), prediction, train.to_numpy()),
                            **timing, clipped_negative=int((raw < 0).sum()),
                            artifact_sha256={name: sha256(task / name) for name in artifacts},
                        )
                    except Exception as exc:
                        record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    write_json(status_path, record)
                    results.append(record)
                    print(fold, target, horizon, model, record["status"], flush=True)
                    pd.DataFrame(results).drop(columns=["artifact_sha256"], errors="ignore").to_csv(
                        root / "metrics.csv", index=False)
    write_json(root / "summary.json", {
        "total": len(results), "success": sum(r["status"] == "success" for r in results),
        "failed": sum(r["status"] != "success" for r in results),
    })
    return str(root)
