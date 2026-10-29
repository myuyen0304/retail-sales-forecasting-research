"""Sinh bảng LaTeX cho paper từ results/tables/*.csv (chạy sau make_tables.py --test).

    python scripts/make_paper_tables.py   → paper/tables/{main_results,ablation,dm_test}.tex

Mọi số là revenue; "Avg" = MAE trung bình 4 chuỗi (đúng luật chọn phương pháp ở protocol §7). Đơn vị: nghìn.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsf import data

TABLES = data.RESULTS / "tables"
OUT = data.ROOT / "paper" / "tables"
SERIES = ["TOTAL", "EAST", "CENTRAL", "WEST"]
SHORT = {"TOTAL": "Total", "EAST": "East", "CENTRAL": "Central", "WEST": "West"}
NAMES = {"snaive1y": "SNaive-1Y", "snaive2y": "SNaive-2Y", "arima": "ARIMA", "sarima": "SARIMA", "sarimax": "SARIMAX",
         "prophet": "Prophet", "rf": "Random Forest", "gb": "Gradient Boosting", "xgboost": "XGBoost",
         "lightgbm": "LightGBM", "rnn": "RNN", "lstm": "LSTM", "ens_avg_topk": "Average of top 3",
         "ens_inv_mae": "Inverse-MAE top 3", "ens_stack_nnls": "NNLS stacking", "sel_online": "Online selection",
         "sarima_w7": "SARIMA $(1,1,1)(1,0,1)_7$", "prophet_pure": "Prophet (default)", "rf_lag": "Random Forest (lags)",
         "gb_lag": "Gradient Boosting (lags)", "xgboost_lag": "XGBoost (lags)", "lightgbm_lag": "LightGBM (lags)",
         "rnn_ar": "RNN (lags)", "lstm_ar": "LSTM (lags)"}
BLOCKS = [("Baselines", ["snaive1y", "snaive2y"]),
          ("Statistical", ["arima", "sarima", "sarimax", "prophet"]),
          ("Machine learning", ["rf", "gb", "xgboost", "lightgbm"]),
          ("Deep learning", ["rnn", "lstm"]),
          ("Combination", ["ens_avg_topk", "ens_inv_mae", "ens_stack_nnls", "sel_online"])]


def k(x: float) -> str:
    return f"{x / 1e3:,.0f}".replace(",", "{,}")


def bold_min(col: pd.Series, fmt) -> list[str]:
    best = col.min()
    return [f"\\textbf{{{fmt(v)}}}" if v == best else fmt(v) for v in col]


def main_results() -> str:
    t1 = pd.read_csv(TABLES / "T1_main.csv")
    rev = t1[t1.target == "revenue"]
    models = [m for _, ms in BLOCKS for m in ms]
    cv = rev.pivot(index="model", columns="series", values="mae_mean").reindex(models)[SERIES]
    te = rev.pivot(index="model", columns="series", values="test_mae").reindex(models)[SERIES]
    mape = rev.pivot(index="model", columns="series", values="test_mape").reindex(models)[SERIES].mean(axis=1)
    cv["Avg"], te["Avg"] = cv.mean(axis=1), te.mean(axis=1)
    # sel_online chỉ có F2–F4 (F1 chưa có fold trước để chọn) → không đặt cạnh trung bình 4 fold
    cv.loc["sel_online"] = float("nan")
    cols = {f"cv_{c}": bold_min(cv[c].dropna(), k) for c in cv}
    cols |= {f"te_{c}": bold_min(te[c], k) for c in te}
    cols["mape"] = bold_min(mape, lambda v: f"{v:.1f}")
    rows, i_cv = [], {m: j for j, m in enumerate(cv[cv.Avg.notna()].index)}
    for title, ms in BLOCKS:
        rows.append(f"\\multicolumn{{12}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for m in ms:
            j = models.index(m)
            c = [cols[f"cv_{s}"][i_cv[m]] if m in i_cv else "--" for s in SERIES + ["Avg"]]
            t = [cols[f"te_{s}"][j] for s in SERIES + ["Avg"]]
            rows.append(" & ".join([NAMES[m]] + c + t + [cols["mape"][j]]) + " \\\\")
    head = " & ".join(SHORT[s] for s in SERIES)
    return "\n".join([
        r"\begin{table*}[t]\centering",
        r"\caption{Revenue forecast accuracy. MAE in thousands: mean over the four CV folds (365-day horizon) and on the test set "
        r"(549-day horizon); Avg is the mean over the four series. Test MAPE (\%) is averaged over the four series. "
        r"Bold: lowest value in each column. Online selection has no F1 forecast and is compared on F2--F4 in the text.}\label{tab:main}",
        r"\footnotesize",
        r"\begin{tabular}{@{}l rrrrr rrrrr r@{}}",
        r"\toprule",
        r" & \multicolumn{5}{c}{Cross-validation MAE} & \multicolumn{5}{c}{Test MAE} & Test \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
        f"Model & {head} & Avg & {head} & Avg & MAPE \\\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])


def ablation() -> str:
    t1 = pd.read_csv(TABLES / "T1_main.csv")
    rev = t1[t1.target == "revenue"]
    avg = lambda col: rev.groupby("model")[col].mean()
    cv, te = avg("mae_mean"), avg("test_mae")
    t2 = pd.read_csv(TABLES / "T2_ablation.csv")
    # T2 có cả cặp (arima, sarimax) nên phải lọc theo bản pure, không chỉ theo bản calendar
    wins = t2[t2.target == "revenue"].groupby(["pure", "calendar"])[["folds_cal_wins", "folds"]].sum()
    pairs = [("sarima_w7", "sarimax"), ("prophet_pure", "prophet"), ("rf_lag", "rf"), ("gb_lag", "gb"),
             ("xgboost_lag", "xgboost"), ("lightgbm_lag", "lightgbm"), ("rnn_ar", "rnn"), ("lstm_ar", "lstm")]
    rows = []
    for pure, cal in pairs:
        g_cv, g_te = 100 * (1 - cv[cal] / cv[pure]), 100 * (1 - te[cal] / te[pure])
        w = wins.loc[(pure, cal)]
        rows.append(f"{NAMES[cal]} & {k(cv[pure])} & {k(cv[cal])} & {g_cv:.0f} & {w.folds_cal_wins}/{w.folds} & "
                    f"{k(te[pure])} & {k(te[cal])} & {g_te:.0f} \\\\")
    # Nhóm biến: CV chỉ F3–F4 (protocol §5), test đủ 549 ngày
    t2b = pd.read_csv(TABLES / "T2b_feature_groups.csv")
    t2b = t2b[t2b.target == "revenue"]
    grp_rows = []
    for base in ("sarimax", "lightgbm"):
        b = t2b[t2b.base == base]
        cells = []
        for g in ("regime", "promo", "dom"):
            x = b[b.dropped == g]
            inc_cv = 100 * (x.mae_variant.mean() / x.mae_base.mean() - 1)
            inc_te = 100 * (te[f"{base}_no{g}"] / te[base] - 1)
            cells += [f"{inc_cv:+.0f}", f"{inc_te:+.0f}"]
        grp_rows.append(f"{NAMES[base]} & " + " & ".join(cells) + " \\\\")
    return "\n".join([
        r"\begin{table}[t]\centering",
        r"\caption{Value of calendar information (revenue, MAE in thousands averaged over the four series). "
        r"Top: pure recursive variant vs.\ calendar-aware model; gain = MAE reduction (\%); wins = folds (of 4 series $\times$ 4 folds) "
        r"in which the calendar-aware model is better. Bottom: MAE change (\%) when one covariate group is removed, on CV folds F3--F4 "
        r"(the folds in which all groups exist) and on the test set.}\label{tab:ablation}",
        r"\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l rrrr rrr@{}}",
        r"\toprule",
        r" & \multicolumn{4}{c}{Cross-validation} & \multicolumn{3}{c}{Test} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
        r"Model & Pure & Cal. & Gain & Wins & Pure & Cal. & Gain \\",
        r"\midrule", *rows,
        r"\midrule",
        r" & \multicolumn{2}{c}{$-$regime} & \multicolumn{2}{c}{$-$promo} & \multicolumn{2}{c}{$-$dom} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r" & CV & Test & CV & Test & CV & Test & \\",
        r"\midrule", *[row.replace(" \\\\", " & \\\\") for row in grp_rows],
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def dm_test() -> str:
    t7 = pd.read_csv(TABLES / "T7_dm_test.csv")
    t7 = t7[t7.target == "revenue"]
    best = t7.groupby("series").best.first()
    show = ["xgboost", "gb", "lightgbm", "ens_stack_nnls", "rnn", "lstm", "rf", "sarimax", "prophet", "snaive1y"]
    p = t7.pivot(index="other", columns="series", values="p_value")
    rows = []
    for m in show:
        cells = []
        for s in SERIES:
            if best[s] == m:
                cells.append("best")
            else:
                v = p.loc[m, s]
                txt = "$<$0.001" if v < 0.001 else f"{v:.3f}"
                cells.append(f"\\textbf{{{txt}}}" if v < 0.05 else txt)
        rows.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    return "\n".join([
        r"\begin{table}[t]\centering",
        r"\caption{Diebold--Mariano tests on the test set (revenue, absolute errors, Newey--West variance, HLN correction): "
        r"$p$-values of each model against the most accurate forecast of the same series (``best''). Bold: $p<0.05$.}\label{tab:dm}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        "Model & " + " & ".join(SHORT[s] for s in SERIES) + " \\\\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("main_results", main_results), ("ablation", ablation), ("dm_test", dm_test)]:
        (OUT / f"{name}.tex").write_text(fn(), encoding="utf-8")
    print("Đã ghi", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
