"""Sinh mọi hình của paper vào results/figures/ (PNG 300 dpi + PDF).

    python scripts/make_figures.py           # hình dữ liệu + CV
    python scripts/make_figures.py --test    # thêm hình test (sau khi chạy stage test)
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rsf import data, results, splits
from rsf.models import ABLATION, BASELINES, MAIN

OUT = results.RESULTS / "figures"
TABLES = results.RESULTS / "tables"

# Màu theo họ model (thứ tự cố định, không theo thứ hạng)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
FAMILY_COLOR = {"baseline": "#8c8b85", "statistical": "#2a78d6", "ml": "#eb6834", "dl": "#1baf7a",
                "ensemble": "#e87ba4", "selection": "#eda100", "pure (ablation)": "#c3c2b7",
                "feature ablation": "#c3c2b7"}
SERIES_COLOR = {"TOTAL": INK, "EAST": "#2a78d6", "CENTRAL": "#eb6834", "WEST": "#1baf7a"}
NAMES = {"snaive1y": "SNaive-1Y", "snaive2y": "SNaive-2Y", "arima": "ARIMA", "sarima": "SARIMA (Fourier)", "sarimax": "SARIMAX",
         "prophet": "Prophet", "rf": "Random Forest", "gb": "Gradient Boosting", "xgboost": "XGBoost",
         "lightgbm": "LightGBM", "rnn": "RNN", "lstm": "LSTM", "ens_avg_topk": "Ens: Avg top-3",
         "ens_inv_mae": "Ens: Inv-MAE", "ens_stack_nnls": "Ens: Stacking", "sel_online": "Online selection",
         "sarima_w7": "SARIMA (s=7)"}

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]   # phân biệt model cùng họ màu

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
                     "legend.frameon": False, "figure.dpi": 100})


def fam(model: str) -> str:
    from rsf.models import REGISTRY
    g = results.family(model)
    if g != "main":
        return g
    return {"statistical": "statistical", "ml": "ml", "dl": "dl"}[REGISTRY[model].family]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def millions(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))


def shade_events(ax, lo, hi):
    """Tô đợt Urban Blowout (tháng 8 năm lẻ) và vạch mốc chế độ 2019, chỉ trong khoảng [lo, hi]."""
    for y in range(lo.year, hi.year + 1):
        a, b = pd.Timestamp(f"{y}-07-30"), pd.Timestamp(f"{y}-09-02")
        if y % 2 == 1 and b >= lo and a <= hi:
            ax.axvspan(max(a, lo), min(b, hi), color="#eda100", alpha=0.18, lw=0)
    if lo <= pd.Timestamp("2019-01-01") <= hi:
        ax.axvline(pd.Timestamp("2019-01-01"), color=MUTED, ls="--", lw=0.8)
    ax.set_xlim(lo, hi)
    loc = mdates.AutoDateLocator(maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


# ---------------------------------------------------------------- hình dữ liệu
def fig_series(df):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, target in zip(axes, data.TARGETS):
        for sid in data.SERIES:
            s = data.series(df, sid, target).rolling(30, center=True).mean()
            ax.plot(s.index, s.values, color=SERIES_COLOR[sid], lw=1.4 if sid == "TOTAL" else 1.0, label=sid)
        shade_events(ax, s.index.min(), s.index.max())
        ax.set_ylabel(f"{target.upper()} (30-day mean)")
        millions(ax)
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axes[1].text(pd.Timestamp("2019-02-01"), axes[1].get_ylim()[1] * 0.92, "regime shift", color=MUTED, fontsize=8)
    save(fig, "F01_series")


def fig_seasonality(df):
    s = data.series(df, "TOTAL", "revenue")
    post = s[s.index >= "2019-01-01"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    m = post.groupby(post.index.month).mean()
    axes[0].bar(m.index, m.values, color="#2a78d6", width=0.7)
    axes[0].set(xlabel="Month", title="Month of year")
    d = post.groupby(post.index.day).mean()
    axes[1].plot(d.index, d.values, color="#2a78d6", marker="o", ms=2.5)
    axes[1].set(xlabel="Day of month", title="Day of month")
    w = post.groupby(post.index.day_of_week).mean()
    axes[2].bar(w.index, w.values, color="#8c8b85", width=0.7)
    axes[2].set(xlabel="Day of week (0=Mon)", title="Day of week")
    for ax in axes:
        millions(ax)
    save(fig, "F02_seasonality")


def fig_splits():
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    rows = [(f.name, splits.START, f.origin, f.end) for f in splits.FOLDS] + [("Test", splits.START, splits.TEST.origin, splits.TEST.end)]
    for i, (name, a, o, e) in enumerate(reversed(rows)):
        ax.barh(i, (o - a).days, left=mdates.date2num(a), color="#b7d3f6", height=0.6)
        ax.barh(i, (e - o).days, left=mdates.date2num(o), color="#eb6834" if name == "Test" else "#2a78d6", height=0.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.xaxis_date()
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#b7d3f6", "#2a78d6", "#eb6834")],
              labels=["training (expanding)", "validation (365 d)", "test (549 d)"], loc="upper left", ncol=3, fontsize=8)
    ax.set_ylim(-0.6, len(rows) + 0.6)
    save(fig, "F03_splits")


# ---------------------------------------------------------------- hình kết quả CV
def fig_cv_box(scores, target="revenue", series="TOTAL"):
    s = scores[(scores.target == target) & (scores.series == series) & scores.model.isin(BASELINES + MAIN + list(NAMES))]
    order = s.groupby("model").mae.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, m in enumerate(order):
        v = s[s.model == m].mae.values
        ax.boxplot(v, positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=FAMILY_COLOR[fam(m)], edgecolor=MUTED, lw=0.6),
                   medianprops=dict(color=INK), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
        ax.scatter(np.full(len(v), i), v, s=8, color=INK, zorder=3)
    ax.set_xticks(range(len(order)), [NAMES.get(m, m) for m in order], rotation=45, ha="right")
    ax.set_ylabel(f"MAE per fold ({series} {target})")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    fams = sorted({fam(m) for m in order}, key=list(FAMILY_COLOR).index)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in fams], labels=fams, ncol=len(fams), loc="upper left")
    save(fig, f"F04_cv_box_{series}_{target}")


def fig_cd():
    path = TABLES / "T3_ranks.csv"
    if not path.exists():
        return
    r = pd.read_csv(path, index_col=0)
    cd = r.nemenyi_cd.iloc[0]
    fig, ax = plt.subplots(figsize=(7.2, 0.35 * len(r) + 0.8))
    y = np.arange(len(r))[::-1]
    ax.scatter(r.avg_rank, y, color=[FAMILY_COLOR[fam(m)] for m in r.index], s=30, zorder=3)
    best = r.avg_rank.min()
    ax.axvspan(best, best + cd, color=GRID, alpha=0.8, lw=0)
    ax.set_yticks(y, [NAMES.get(m, m) for m in r.index])
    ax.set_xlabel(f"Average rank (lower is better); shaded = within Nemenyi CD={cd:.2f} of best; Friedman p={r.friedman_p.iloc[0]:.1e}")
    save(fig, "F05_cd_ranks")


def fig_ablation():
    path = TABLES / "T2_ablation.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    t = t[t.target == "revenue"].groupby(["pure", "calendar"]).gain_pct.agg(["mean", "min", "max"]).reset_index()
    t = t.sort_values("mean")
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    y = np.arange(len(t))
    ax.barh(y, t["mean"], color="#2a78d6", height=0.6)
    ax.errorbar(t["mean"], y, xerr=[t["mean"] - t["min"], t["max"] - t["mean"]], fmt="none", ecolor=INK, lw=0.8, capsize=2)
    ax.set_yticks(y, [f"{NAMES.get(c, c)} vs {p}" for p, c in zip(t.pure, t.calendar)])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("MAE reduction from calendar features, % (mean and range over 4 series)")
    save(fig, "F06_ablation")


def fig_overlay(preds, split, name, top=3, series="TOTAL", target="revenue"):
    p = preds[(preds.series == series) & (preds.target == target) & (preds.split == split)]
    if p.empty:
        return
    maes = p.groupby("model").apply(lambda g: (g.y_true - g.y_pred).abs().mean())
    chosen = list(maes[maes.index.isin(MAIN + list(NAMES))].nsmallest(top).index) + ["snaive1y"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    truth = p.groupby("date").y_true.first()
    roll = lambda v: pd.Series(v).rolling(7, center=True, min_periods=1).mean().values
    ax.plot(truth.index, truth.values, color="#dddcd6", lw=0.6, label="Actual (daily)")
    ax.plot(truth.index, roll(truth.values), color=INK, lw=1.0, label="Actual (7-day mean)")
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8c8b85"]
    for m, c in zip(chosen, colors):
        g = p[p.model == m].sort_values("date")
        if not g.empty:
            ax.plot(g.date, roll(g.y_pred.values), color=c, lw=1.1, label=f"{NAMES.get(m, m)} (MAE {maes[m] / 1e3:.0f}k)")
    shade_events(ax, truth.index.min(), truth.index.max())
    millions(ax)
    ax.set_ylabel(f"{series} {target} (forecasts: 7-day mean)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
    save(fig, name)


def fig_lead(tag):
    path = TABLES / f"T6_lead_{tag}.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    t = t[(t.series == "TOTAL") & (t.target == "revenue") & t.model.isin(BASELINES + MAIN)]
    if t.empty:
        return
    piv = t.pivot_table(index="bucket", columns="model", values="mae").dropna(how="all")
    piv = piv.reindex([b for b in [f"{a}-{b}" for a, b in splits.LEAD_BUCKETS] if b in piv.index])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for m in piv.columns:
        ax.plot(range(len(piv)), piv[m].values, marker=MARKERS[list(piv.columns).index(m) % len(MARKERS)], ms=4,
                color=FAMILY_COLOR[fam(m)], lw=1.0, alpha=0.9, label=NAMES.get(m, m))
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7.5)
    ax.set_xticks(range(len(piv)), [f"{b} d" for b in piv.index])
    ax.set_xlabel("Lead time")
    ax.set_ylabel("MAE (TOTAL revenue)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    save(fig, f"F08_lead_{tag}")


def fig_bottomup():
    path = TABLES / "T4_bottomup.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    t = t[(t.target == "revenue") & t.model.isin(MAIN)].sort_values("gain_pct")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.barh(range(len(t)), t.gain_pct, color=[FAMILY_COLOR[fam(m)] for m in t.model], height=0.6)
    ax.set_yticks(range(len(t)), [NAMES.get(m, m) for m in t.model])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("MAE reduction of bottom-up (sum of 3 regions) vs direct TOTAL forecast, %")
    save(fig, "F09_bottomup")


def fig_train_val_test(test_preds, model, series="TOTAL", target="revenue"):
    """Mục 3.4: actual vs predicted trên train / validation (F4) / test cho một model."""
    df, _ = data.load()
    y = data.series(df, series, target)
    fit_path = results.RESULTS / "preds" / "test" / model / f"{series}_{target}_TEST_s{42 if model not in ('rnn', 'lstm') else 0}_fitted.parquet"
    cv = results.load_preds("cv")
    v = cv[(cv.model == model) & (cv.series == series) & (cv.target == target) & (cv.split == "F4")]
    t = test_preds[(test_preds.model == model) & (test_preds.series == series) & (test_preds.target == target)]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.plot(y.index, y.rolling(7, center=True).mean(), color="#c3c2b7", lw=0.8, label="Actual (7-day mean)")
    if fit_path.exists():
        f = pd.read_parquet(fit_path)
        f = f[f.date <= splits.TRAIN_END]
        ax.plot(f.date, f.y_fit.rolling(7, center=True).mean(), color="#2a78d6", lw=0.9, label="Fitted (train)")
    ax.plot(v.date, v.y_pred.rolling(7, center=True).mean(), color="#1baf7a", lw=1.0, label="Forecast (validation)")
    ax.plot(t.date, t.y_pred.rolling(7, center=True).mean(), color="#eb6834", lw=1.0, label="Forecast (test)")
    for x in (splits.TRAIN_END, splits.VAL_END):
        ax.axvline(x, color=MUTED, ls="--", lw=0.8)
    ax.set_xlim(pd.Timestamp("2016-01-01"), splits.TEST_END)
    millions(ax)
    ax.set_title(f"{NAMES.get(model, model)}: {series} {target}", fontsize=9)
    ax.legend(ncol=4, loc="upper right", fontsize=7.5)
    save(fig, f"F10_train_val_test_{model}")


def main(with_test):
    df, _ = data.load()
    fig_series(df)
    fig_seasonality(df)
    fig_splits()
    s_path = TABLES / "scores_cv_by_fold.csv"
    if s_path.exists():
        scores = pd.read_csv(s_path)
        for sid in ("TOTAL",):
            for target in data.TARGETS:
                fig_cv_box(scores, target, sid)
        fig_cd()
        fig_ablation()
        fig_bottomup()
        cv = results.load_preds("cv")
        cv_ens, _, _ = results.add_ensembles(cv)
        fig_overlay(pd.concat([cv, cv_ens]), "F4", "F07_validation_overlay")
        fig_lead("cv")
    if with_test:
        test = results.load_preds("test")
        cv = results.load_preds("cv")
        _, test_ens, _ = results.add_ensembles(cv, test)
        allt = pd.concat([test, test_ens])
        fig_overlay(allt, "TEST", "F11_test_overlay")
        fig_lead("test")
        best = pd.read_csv(TABLES / "scores_test.csv").query("series == 'TOTAL' and target == 'revenue' and model in @MAIN").sort_values("test_mae").model.iloc[0]
        fig_train_val_test(test, best)
    print("Đã ghi", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    main(ap.parse_args().test)
