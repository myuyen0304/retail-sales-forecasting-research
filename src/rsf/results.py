"""Đọc dự báo đã lưu và dựng các bảng phái sinh (trung bình seed, ensemble, bottom-up, COGS qua tỷ lệ).

Mọi bảng/hình của paper đi qua module này, để một con số chỉ có một nguồn.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rsf import data, ensemble, metrics, splits
from rsf.features import Calendar
from rsf.models import BASELINES, FEATURE_ABLATION, MAIN, PURE

RESULTS = data.RESULTS
KEYS = ["model", "series", "target", "split"]


def load_preds(stage: str) -> pd.DataFrame:
    """Dự báo dạng dài; model nhiều seed được lấy trung bình theo ngày (cột n_seeds ghi số seed)."""
    root = RESULTS / "preds" / stage
    frames = []
    for path in root.glob("*/*.parquet"):
        if path.stem.endswith("_fitted"):
            continue
        sid, target, split, seed = path.stem.split("_")
        frames.append(pd.read_parquet(path).assign(model=path.parent.name, series=sid, target=target,
                                                   split=split, seed=int(seed[1:])))
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    agg = (raw.groupby(KEYS + ["date", "lead"], as_index=False)
              .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"), n_seeds=("seed", "nunique")))
    return agg


def seed_scores(stage: str) -> pd.DataFrame:
    """Điểm từng seed (để báo mean ± sd của RNN/LSTM)."""
    log = pd.read_csv(RESULTS / f"runs_{stage}.csv")
    return log[log.status == "ok"]


def score_table(preds: pd.DataFrame, by=KEYS) -> pd.DataFrame:
    return (preds.groupby(by)
                 .apply(lambda g: pd.Series(metrics.scores(g.y_true, g.y_pred)), include_groups=False)
                 .reset_index())


# ---------------------------------------------------------------- ensemble
def add_ensembles(cv: pd.DataFrame, test: pd.DataFrame | None = None, candidates=MAIN):
    """Trả (cv_ens, test_ens). Fold Fk học trọng số từ các fold trước; test học từ cả 4 fold."""
    fold_order = [f.name for f in splits.FOLDS]
    out_cv, out_test, used = [], [], []
    for (sid, target), g in cv[cv.model.isin(candidates)].groupby(["series", "target"]):
        cands = [m for m in candidates if m in set(g.model)]
        # Đối chứng công bằng cho RQ3: một model đơn chọn bằng cùng luật online (MAE thấp nhất trên các fold trước)
        for i, fold in enumerate(fold_order[1:], start=1):
            hist = g[g.split.isin(fold_order[:i])]
            if hist.empty:
                continue
            best = hist.groupby("model").apply(lambda x: metrics.mae(x.y_true, x.y_pred)).idxmin()
            cur = g[(g.split == fold) & (g.model == best)]
            out_cv.append(cur[["date", "y_true", "y_pred"]].assign(model="sel_online", series=sid, target=target, split=fold))
            used.append(dict(method="sel_online", series=sid, target=target, split=fold, weights=json.dumps({best: 1.0})))
        if test is not None and not test.empty:
            best = g.groupby("model").apply(lambda x: metrics.mae(x.y_true, x.y_pred)).idxmin()
            cur = test[(test.series == sid) & (test.target == target) & (test.model == best)]
            out_test.append(cur[["date", "y_true", "y_pred"]].assign(model="sel_online", series=sid, target=target, split="TEST"))
            used.append(dict(method="sel_online", series=sid, target=target, split="TEST", weights=json.dumps({best: 1.0})))
        for method in ensemble.METHODS:
            for i, fold in enumerate(fold_order):
                hist = g[g.split.isin(fold_order[:i])]
                w = ensemble.weights(method, hist, cands)
                cur = g[g.split == fold]
                if not set(w) <= set(cur.model):
                    continue
                c = ensemble.combine(cur[cur.model.isin(w)], w)
                out_cv.append(c.assign(model=f"ens_{method}", series=sid, target=target, split=fold))
                used.append(dict(method=method, series=sid, target=target, split=fold, weights=json.dumps(w)))
            if test is not None and not test.empty:
                w = ensemble.weights(method, g, cands)
                cur = test[(test.series == sid) & (test.target == target) & test.model.isin(w)]
                if set(w) <= set(cur.model):
                    c = ensemble.combine(cur, w)
                    out_test.append(c.assign(model=f"ens_{method}", series=sid, target=target, split="TEST"))
                    used.append(dict(method=method, series=sid, target=target, split="TEST", weights=json.dumps(w)))
    lead = lambda d: d.assign(lead=d.groupby(KEYS).cumcount() + 1)
    cv_ens = lead(pd.concat(out_cv)) if out_cv else pd.DataFrame()
    test_ens = lead(pd.concat(out_test)) if out_test else pd.DataFrame()
    return cv_ens, test_ens, pd.DataFrame(used)


# ---------------------------------------------------------------- multi-region
def bottom_up(preds: pd.DataFrame) -> pd.DataFrame:
    """Cộng dự báo 3 vùng của cùng model → chuỗi BOTTOMUP, so với TOTAL."""
    reg = preds[preds.series.isin(data.REGIONS)]
    ok = reg.groupby(["model", "target", "split"]).series.nunique() == 3
    reg = reg.set_index(["model", "target", "split"]).loc[ok[ok].index].reset_index()
    s = reg.groupby(["model", "target", "split", "date", "lead"], as_index=False)["y_pred"].sum()
    truth = preds[preds.series == "TOTAL"][["target", "split", "date", "y_true"]].drop_duplicates()
    return s.merge(truth, on=["target", "split", "date"]).assign(series="BOTTOMUP")


# ---------------------------------------------------------------- COGS qua tỷ lệ
def cogs_via_ratio(preds: pd.DataFrame) -> pd.DataFrame:
    """COGS = revenue dự báo × tỷ lệ ΣCOGS/ΣRevenue theo (tháng, cờ Urban Blowout) học trên train của split."""
    df, promos = data.load()
    out = []
    for (model, sid, split), g in preds[preds.target == "revenue"].groupby(["model", "series", "split"]):
        origin = splits.SPLITS[split].origin
        tr = df[(df.series_id == sid) & (df.date <= origin)].set_index("date")
        cal = Calendar(promos, origin)
        ub_col = next((c for c in cal.rules.slug if "urban" in c), None)
        flag = lambda d: (cal.promo_flags(pd.DatetimeIndex(d))[f"promo_{ub_col}"].values if ub_col else np.zeros(len(d)))
        key_tr = pd.MultiIndex.from_arrays([tr.index.month, flag(tr.index)])
        prof = tr.groupby(key_tr)[["cogs", "revenue"]].sum()
        ratio = prof.cogs / prof.revenue
        by_month = tr.groupby(tr.index.month)[["cogs", "revenue"]].sum()
        by_month = by_month.cogs / by_month.revenue
        d = pd.DatetimeIndex(g.date)
        r = [ratio.get((m, u), by_month[m]) for m, u in zip(d.month, flag(d))]
        truth = df[(df.series_id == sid)].set_index("date").cogs.reindex(d).values
        out.append(pd.DataFrame({"model": model, "series": sid, "target": "cogs_ratio", "split": split,
                                 "date": d, "lead": g.lead.values, "y_true": truth, "y_pred": g.y_pred.values * r}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def family(model: str) -> str:
    if model.startswith("ens_"):
        return "ensemble"
    if model == "sel_online":
        return "selection"
    if model in FEATURE_ABLATION:
        return "feature ablation"
    if model in BASELINES:
        return "baseline"
    if model in PURE:
        return "pure (ablation)"
    return "main"
