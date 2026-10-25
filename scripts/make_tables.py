"""Sinh mọi bảng của paper vào results/tables/ từ dự báo đã lưu.

    python scripts/make_tables.py            # chỉ CV
    python scripts/make_tables.py --test     # thêm cột test (chỉ chạy sau khi đã khóa giao thức và chạy stage test)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rsf import data, metrics, results, splits
from rsf.models import ABLATION, BASELINES, FEATURE_ABLATION, MAIN, REGISTRY

OUT = results.RESULTS / "tables"


def mean_sd(t: pd.DataFrame, by: list[str], col: str) -> pd.DataFrame:
    g = t.groupby(by)[col]
    return pd.DataFrame({f"{col}_mean": g.mean(), f"{col}_sd": g.std()})


def lead_bucket(lead: pd.Series) -> pd.Series:
    labels = [f"{a}-{b}" for a, b in splits.LEAD_BUCKETS]
    return pd.cut(lead, [0] + [b for _, b in splits.LEAD_BUCKETS], labels=labels)


def main(with_test: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    cv = results.load_preds("cv")
    test = results.load_preds("test") if with_test else pd.DataFrame()
    cv_ens, test_ens, w = results.add_ensembles(cv, test if with_test else None)
    w.to_csv(OUT / "ensemble_weights.csv", index=False)
    cv_all = pd.concat([cv, cv_ens], ignore_index=True)
    test_all = pd.concat([test, test_ens], ignore_index=True) if with_test else pd.DataFrame()

    # T1: bảng chính, CV mean ± sd theo fold, từng chuỗi và target
    s_cv = results.score_table(cv_all)
    s_cv.to_csv(OUT / "scores_cv_by_fold.csv", index=False)
    t1 = pd.concat([mean_sd(s_cv, ["series", "target", "model"], c) for c in ("mae", "rmse", "mape")], axis=1).reset_index()
    # RQ3 so trên F2–F4: ở F1 ensemble/selection chưa có fold trước để học
    late = s_cv[s_cv.split.isin(["F2", "F3", "F4"])].groupby(["series", "target", "model"]).mae.mean().rename("mae_F2F4")
    t1 = t1.merge(late.reset_index(), on=["series", "target", "model"], how="left")
    if with_test:
        s_test = results.score_table(test_all).drop(columns="split").rename(columns={c: f"test_{c}" for c in ("mae", "rmse", "mape")})
        t1 = t1.merge(s_test, on=["series", "target", "model"], how="left")
        s_test.to_csv(OUT / "scores_test.csv", index=False)
    t1["group"] = t1.model.map(results.family)
    t1.sort_values(["series", "target", "mae_mean"]).to_csv(OUT / "T1_main.csv", index=False)

    # T2: ablation calendar-aware vs pure, số fold bản calendar thắng
    rows = []
    for pure, cal in ABLATION.items():
        a = s_cv[s_cv.model == pure].set_index(["series", "target", "split"]).mae
        b = s_cv[s_cv.model == cal].set_index(["series", "target", "split"]).mae
        j = pd.concat([a.rename("mae_pure"), b.rename("mae_cal")], axis=1).dropna()
        for (sid, tgt), g in j.groupby(level=[0, 1]):
            rows.append(dict(pure=pure, calendar=cal, series=sid, target=tgt, mae_pure=g.mae_pure.mean(),
                             mae_cal=g.mae_cal.mean(), gain_pct=100 * (1 - g.mae_cal.mean() / g.mae_pure.mean()),
                             folds_cal_wins=int((g.mae_cal < g.mae_pure).sum()), folds=len(g)))
    pd.DataFrame(rows).to_csv(OUT / "T2_ablation.csv", index=False)

    # T2b: ablation theo nhóm biến — tăng MAE (%) khi bỏ từng nhóm khỏi model gốc.
    # Chỉ tính trên F3–F4: ở F1–F2 chưa có `post2019` nên bản *_noregime trùng bản gốc (làm loãng hiệu ứng).
    by_fold = s_cv[s_cv.model.isin(FEATURE_ABLATION + sorted({REGISTRY[v].base for v in FEATURE_ABLATION}))]
    if not by_fold.empty:
        by_fold.to_csv(OUT / "T2b_feature_groups_by_fold.csv", index=False)
    rows = []
    for v in FEATURE_ABLATION:
        base = REGISTRY[v].base
        sub = s_cv[s_cv.model.isin([v, base]) & s_cv.split.isin(["F3", "F4"])]
        j = sub.pivot_table(index=["series", "target", "split"], columns="model", values="mae").dropna()
        for (sid, tgt), g in j.groupby(level=[0, 1]):
            rows.append(dict(variant=v, base=base, dropped=v.split("_no")[-1], series=sid, target=tgt,
                             mae_base=g[base].mean(), mae_variant=g[v].mean(),
                             mae_increase_pct=100 * (g[v].mean() / g[base].mean() - 1),
                             folds_base_wins=int((g[base] < g[v]).sum()), folds=len(g)))
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "T2b_feature_groups.csv", index=False)

    # T3: Friedman + Nemenyi, revenue. (a) model chính + baseline, 4 chuỗi × 4 fold;
    # (b) thêm ensemble và selection online, chỉ F2–F4 (F1 chưa có fold trước để học trọng số)
    ens = [f"ens_{m}" for m in ("avg_topk", "inv_mae", "stack_nnls")] + ["sel_online"]
    for name, cand, folds in [("T3_ranks", BASELINES + MAIN, ["F1", "F2", "F3", "F4"]),
                              ("T3b_ranks_F2F4", BASELINES + MAIN + ens, ["F2", "F3", "F4"])]:
        sub = s_cv[(s_cv.target == "revenue") & s_cv.model.isin(cand) & s_cv.split.isin(folds)]
        piv = sub.pivot_table(index=["series", "split"], columns="model", values="mae").dropna(axis=1)
        if piv.shape[1] >= 3 and len(piv) >= 2:
            ranks, p, cd = metrics.friedman_nemenyi(piv)
            ranks.rename("avg_rank").to_frame().assign(friedman_p=p, nemenyi_cd=cd, n_problems=len(piv)).to_csv(OUT / f"{name}.csv")

    # T4: bottom-up (Σ 3 vùng) vs TOTAL trực tiếp
    bu = results.bottom_up(cv)
    if not bu.empty:
        sb = results.score_table(bu)
        direct = s_cv[s_cv.series == "TOTAL"]
        t4 = sb.merge(direct, on=["model", "target", "split"], suffixes=("_bottomup", "_direct"))
        t4.groupby(["model", "target"])[["mae_bottomup", "mae_direct"]].mean().assign(
            gain_pct=lambda d: 100 * (1 - d.mae_bottomup / d.mae_direct)).to_csv(OUT / "T4_bottomup.csv")

    # T5: COGS trực tiếp vs qua tỷ lệ
    cr = results.cogs_via_ratio(cv[cv.model.isin(MAIN + BASELINES)])
    if not cr.empty:
        sr = results.score_table(cr)
        t5 = sr.merge(s_cv[s_cv.target == "cogs"], on=["model", "series", "split"], suffixes=("_ratio", "_direct"))
        t5.groupby(["series", "model"])[["mae_ratio", "mae_direct"]].mean().to_csv(OUT / "T5_cogs_ratio.csv")

    # T6: sai số theo lead time và tháng 8 năm lẻ
    base = test_all if with_test else cv_all
    tag = "test" if with_test else "cv"
    b = base.assign(bucket=lead_bucket(base.lead))
    results.score_table(b, by=["series", "target", "model", "bucket"]).to_csv(OUT / f"T6_lead_{tag}.csv", index=False)
    d = pd.to_datetime(base.date)
    aug = base[(d.dt.month == 8) & (d.dt.year % 2 == 1)]
    if not aug.empty:
        results.score_table(aug, by=["series", "target", "model", "split"]).to_csv(OUT / f"T6_odd_august_{tag}.csv", index=False)

    # T7: Diebold–Mariano ở test, mỗi model so với model tốt nhất
    if with_test:
        rows = []
        for (sid, tgt), g in test_all.groupby(["series", "target"]):
            maes = g.groupby("model").apply(lambda x: metrics.mae(x.y_true, x.y_pred))
            best = maes.idxmin()
            eb = g[g.model == best].sort_values("date")
            for m in maes.index.drop(best):
                em = g[g.model == m].sort_values("date")
                stat, p = metrics.diebold_mariano(eb.y_true.values - eb.y_pred.values, em.y_true.values - em.y_pred.values)
                rows.append(dict(series=sid, target=tgt, best=best, other=m, dm_stat=stat, p_value=p))
        pd.DataFrame(rows).to_csv(OUT / "T7_dm_test.csv", index=False)

    # T8: thời gian chạy
    log = pd.read_csv(results.RESULTS / "runs_cv.csv")
    log[log.status == "ok"].groupby("model")[["fit_s", "predict_s"]].median().to_csv(OUT / "T8_runtime.csv")
    key = ["model", "series", "target", "split", "seed"]
    ok_keys = set(map(tuple, log.loc[log.status == "ok", key].values))
    fails = log[(log.status != "ok") & ~log[key].apply(tuple, axis=1).isin(ok_keys)]   # bỏ lỗi đã chạy lại thành công
    if len(fails):
        print(f"CẢNH BÁO: {len(fails)} task lỗi, xem results/runs_cv.csv")
    print("Đã ghi", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    main(ap.parse_args().test)
