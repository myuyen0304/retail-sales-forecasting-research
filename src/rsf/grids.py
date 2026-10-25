"""Lưới tham số (tối đa 12 cấu hình mỗi model), đóng băng trong protocol.md trước khi chạy.

Tinh chỉnh trên chuỗi TOTAL / revenue, 4 fold CV, chọn theo MAE trung bình; cấu hình thắng dùng cho mọi chuỗi và target.
Model không có trong đây chạy với tham số mặc định (baseline, ARIMA tự chọn bậc bằng AIC, các bản pure).
"""
import itertools


def _product(**axes):
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


ARMA = [(1, 0, 1), (2, 0, 1), (1, 0, 2), (2, 0, 2)]

GRIDS = {
    "sarima": _product(order=ARMA),
    "sarimax": _product(order=ARMA),
    "prophet": _product(growth=["flat", "linear"], changepoint_prior_scale=[0.01, 0.05, 0.5]),
    "rf": _product(min_samples_leaf=[1, 3, 10], max_features=[0.33, 0.5, 1.0]),
    "gb": _product(learning_rate=[0.03, 0.1], max_leaf_nodes=[7, 15, 31]),
    "xgboost": _product(learning_rate=[0.03, 0.1], max_depth=[3, 4, 6]),
    "lightgbm": _product(learning_rate=[0.03, 0.1], num_leaves=[7, 15, 31], min_child_samples=[10, 30]),
    "rnn": _product(hidden=[16, 32, 64], lookback=[14, 28, 56]),
    "lstm": _product(hidden=[16, 32, 64], lookback=[14, 28, 56]),
}

assert all(len(g) <= 12 for g in GRIDS.values())
