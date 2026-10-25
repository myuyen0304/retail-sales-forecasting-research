"""Chạy thực nghiệm. Ba giai đoạn, mỗi task ghi một file nên chạy lại sẽ bỏ qua task đã xong:

  python -m rsf.run tune --models lightgbm,lstm     # lưới tham số, TOTAL/revenue, 4 fold → results/best_params.json
  python -m rsf.run cv   --models all               # cấu hình đã chọn, 4 chuỗi × 2 target × 4 fold
  python -m rsf.run test --models all               # fit tới 2021-06-30, chấm 2021-07-01 → 2022-12-31 (một lần)
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rsf import data, metrics, splits
from rsf.features import Calendar
from rsf.grids import GRIDS
from rsf.models import BASELINES, FEATURE_ABLATION, MAIN, PURE, REGISTRY

RESULTS = data.RESULTS
SEEDS = [0, 1, 2, 3, 4]


def write_env(stage: str) -> None:
    """Ghi phiên bản Python/thư viện/máy của lượt chạy (laptop và Colab có thể khác nhau)."""
    import importlib.metadata as md
    import os
    import platform
    pkgs = {}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "lightgbm", "xgboost", "prophet", "torch"]:
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    env = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "colab": "COLAB_RELEASE_TAG" in os.environ, "packages": pkgs,
           "time": pd.Timestamp.now().isoformat(timespec="seconds")}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"env_{stage}.json"
    history = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(history + [env], indent=2))


def run_task(model: str, params: dict, seed: int, y: pd.Series, promos: pd.DataFrame, split: splits.Split,
             with_fitted: bool = False) -> tuple[pd.DataFrame, dict, pd.Series | None]:
    train = y.loc[:split.origin]
    dates = split.dates
    cal = Calendar(promos, split.origin)
    m = REGISTRY[model](seed=seed, **params)
    t0 = time.perf_counter()
    m.fit(train, cal)
    t1 = time.perf_counter()
    pred = m.predict(dates)
    t2 = time.perf_counter()
    if pred.shape != (len(dates),) or not np.isfinite(pred).all():
        raise ValueError("Dự báo sai độ dài hoặc có NaN/Inf")
    truth = y.reindex(dates).values
    out = pd.DataFrame({"date": dates, "lead": np.arange(1, len(dates) + 1), "y_true": truth, "y_pred": pred})
    meta = {"fit_s": round(t1 - t0, 3), "predict_s": round(t2 - t1, 3), **metrics.scores(truth, pred)}
    for attr in ("converged", "selected_order", "epochs"):
        if hasattr(m, attr):
            meta[attr] = str(getattr(m, attr))
    return out, meta, (m.fitted() if with_fitted else None)


def _models(arg: str) -> list[str]:
    if arg == "all":
        return BASELINES + MAIN + PURE + FEATURE_ABLATION
    groups = {"main": MAIN, "pure": PURE, "baselines": BASELINES, "ablation": FEATURE_ABLATION}
    out = []
    for a in arg.split(","):
        out += groups.get(a, [a])
    unknown = set(out) - REGISTRY.keys()
    if unknown:
        raise SystemExit(f"Model không có: {unknown}")
    return out


def _seeds(model: str, n: int) -> list[int]:
    return SEEDS[:n] if REGISTRY[model].stochastic else [42]


def _safe_task(key, model, params, seed, y, promos, split, with_fitted=False):
    """Chạy trong tiến trình con; lỗi được trả về thay vì làm dừng cả lượt."""
    try:
        return key, run_task(model, params, seed, y, promos, split, with_fitted), None
    except Exception as e:
        return key, None, (repr(e), traceback.format_exc()[-800:])


def _parallel(jobs: int, calls):
    if jobs == 1:
        for c in calls:
            yield c[0](*c[1:])
        return
    yield from Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(c[0])(*c[1:]) for c in calls)


def tune(models, folds, n_seeds=1, jobs=1):
    df, promos = data.load()
    y = data.series(df, "TOTAL", "revenue")
    out_dir = RESULTS / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = RESULTS / "best_params.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    for model in models:
        grid = GRIDS.get(model)
        if not grid:
            print(f"{model}: không có lưới, dùng mặc định")
            continue
        path = out_dir / f"{model}.csv"
        done = pd.read_csv(path) if path.exists() else pd.DataFrame()
        rows = [r for r in done.to_dict("records") if pd.notna(r.get("mae"))]   # task lỗi được chạy lại
        have = {(r["config"], r["fold"], r["seed"]) for r in rows}
        calls = [(_safe_task, (cid, f.name, seed), model, params, seed, y, promos, f)
                 for cid, params in enumerate(grid) for f in folds for seed in _seeds(model, n_seeds)
                 if (cid, f.name, seed) not in have]
        for (cid, fold, seed), res, err in _parallel(jobs, calls):
            row = {"config": cid, "params": json.dumps(grid[cid]), "fold": fold, "seed": seed}
            row.update(res[1] if res else {"error": err[0]})   # cấu hình lỗi không được chấm
            rows.append(row)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(model, cid, fold, seed, row.get("mae", row.get("error")), flush=True)
        t = pd.DataFrame(rows)
        ok = t.groupby("config").filter(lambda g: g["mae"].notna().all() and len(g) == len(folds) * len(_seeds(model, n_seeds)))
        if ok.empty:
            print(f"{model}: không cấu hình nào chạy đủ fold")
            continue
        cid = ok.groupby("config")["mae"].mean().idxmin()
        # đọc lại ngay trước khi ghi, để không xóa mục của lượt tune khác chạy song song
        best = json.loads(best_path.read_text()) if best_path.exists() else {}
        best[model] = json.loads(ok.loc[ok.config == cid, "params"].iloc[0])
        best_path.write_text(json.dumps(best, indent=2))
        print(f"{model}: chọn cấu hình {cid} = {best[model]}")


def evaluate(stage, models, series_ids, targets, split_list, n_seeds, jobs=1):
    df, promos = data.load()
    best_path = RESULTS / "best_params.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    log_path = RESULTS / f"runs_{stage}.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    calls = []
    for model in models:
        base = getattr(REGISTRY[model], "base", model)   # biến thể ablation dùng tham số đã tune của model gốc
        params = best.get(base, {})
        if base in GRIDS and base not in best:
            print(f"CẢNH BÁO: {model} chưa tune, dùng tham số mặc định")
        for sid in series_ids:
            for target in targets:
                y = data.series(df, sid, target)
                for sp in split_list:
                    for seed in _seeds(model, n_seeds):
                        path = RESULTS / "preds" / stage / model / f"{sid}_{target}_{sp.name}_s{seed}.parquet"
                        if not path.exists():
                            key = (model, sid, target, sp.name, seed, str(path), json.dumps(params))
                            calls.append((_safe_task, key, model, params, seed, y, promos, sp, stage == "test"))
    print(f"{len(calls)} task cần chạy", flush=True)
    for (model, sid, target, split, seed, path, params), res, err in _parallel(jobs, calls):
        path = Path(path)
        rec = {"stage": stage, "model": model, "series": sid, "target": target, "split": split,
               "seed": seed, "params": params}
        if res:
            pred, meta, fitted = res
            path.parent.mkdir(parents=True, exist_ok=True)
            if fitted is not None:
                pd.DataFrame({"date": fitted.index, "y_fit": fitted.values}).to_parquet(
                    path.with_name(path.stem + "_fitted.parquet"))
            pred.to_parquet(path)
            rec.update(meta, status="ok")
        else:
            rec.update(status="failed", error=err[0], trace=err[1])
        log.append(rec)
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(stage, model, sid, target, split, seed, rec.get("mae", rec.get("error")), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tune", "cv", "test"])
    ap.add_argument("--models", default="all", help="all | main | pure | baselines | tên,tên")
    ap.add_argument("--series", default=",".join(data.SERIES))
    ap.add_argument("--targets", default=",".join(data.TARGETS))
    ap.add_argument("--folds", default="F1,F2,F3,F4")
    ap.add_argument("--seeds", type=int, default=5, help="số seed cho model ngẫu nhiên (RNN/LSTM)")
    ap.add_argument("--jobs", type=int, default=1, help="số tiến trình song song")
    a = ap.parse_args(argv)
    models = _models(a.models)
    write_env(a.stage)
    folds = [splits.SPLITS[f] for f in a.folds.split(",")]
    if a.stage == "tune":
        tune(models, folds, n_seeds=1, jobs=a.jobs)
    elif a.stage == "cv":
        evaluate("cv", models, a.series.split(","), a.targets.split(","), folds, a.seeds, a.jobs)
    else:
        evaluate("test", models, a.series.split(","), a.targets.split(","), [splits.TEST], a.seeds, a.jobs)


if __name__ == "__main__":
    main()
