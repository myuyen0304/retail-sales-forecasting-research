"""Danh mục model. `MAIN`: bản chính để xếp hạng; `PURE`: bản không biến lịch cho ablation RQ2."""
from rsf.models.baselines import SNaive1Y, SNaive2Y
from rsf.models.neural import LSTM, LSTM_AR, RNN, RNN_AR
from rsf.models.stats import ARIMA, SARIMA, SARIMAX, ProphetCal, ProphetPure, SARIMAWeekly
from rsf.models.trees import GB, GB_LAG, LGBM, LGBM_LAG, RF, RF_LAG, XGB, XGB_LAG

# Ablation theo nhóm biến (RQ2): bỏ một nhóm, giữ tham số đã tune của model gốc (`base`)
GROUP_ABLATION = {}
for _base in (SARIMAX, LGBM):
    for _g in ("regime", "promo", "dom"):
        _cls = type(f"{_base.__name__}_no_{_g}", (_base,),
                    {"name": f"{_base.name}_no{_g}", "drop_groups": (_g,), "base": _base.name})
        GROUP_ABLATION[_cls.name] = _cls

REGISTRY = {m.name: m for m in list(GROUP_ABLATION.values()) + [
    SNaive1Y, SNaive2Y,
    ARIMA, SARIMAWeekly, SARIMA, SARIMAX, ProphetPure, ProphetCal,
    RF, GB, XGB, LGBM, RF_LAG, GB_LAG, XGB_LAG, LGBM_LAG,
    RNN, LSTM, RNN_AR, LSTM_AR,
]}

BASELINES = ["snaive1y", "snaive2y"]
MAIN = ["arima", "sarima", "sarimax", "prophet", "rf", "gb", "xgboost", "lightgbm", "rnn", "lstm"]
FEATURE_ABLATION = list(GROUP_ABLATION)
PURE = ["sarima_w7", "prophet_pure", "rf_lag", "gb_lag", "xgboost_lag", "lightgbm_lag", "rnn_ar", "lstm_ar"]
# Cặp ablation (bản pure, bản calendar-aware) cho RQ2
ABLATION = {"sarima_w7": "sarimax", "prophet_pure": "prophet", "rf_lag": "rf", "gb_lag": "gb",
            "xgboost_lag": "xgboost", "lightgbm_lag": "lightgbm", "rnn_ar": "rnn", "lstm_ar": "lstm",
            "arima": "sarimax"}
