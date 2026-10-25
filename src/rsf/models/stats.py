"""ARIMA, SARIMA, SARIMAX và Prophet, đều fit trên log(y).

- `arima`: không biến ngoài, bậc chọn bằng AIC trong lưới nhỏ (pure).
- `sarima_w7`: SARIMA mùa vụ tuần s=7 (cấu hình của pilot cũ, pure ablation).
- `sarima`: mùa vụ năm bằng Fourier + sai số ARMA (dynamic harmonic regression), cách khả thi cho s=365.
- `sarimax`: Fourier + toàn bộ biến lịch/khuyến mãi/chế độ + sai số ARMA.
- `prophet_pure` / `prophet`: mặc định / có biến lịch làm regressor.
"""
from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd

from rsf.models.base import Forecaster

MAXITER = 1000


def _fit_sarimax(y, exog, order, seasonal_order=(0, 0, 0, 0), trend="n"):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(y, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend)
        start = None
        if exog is not None and seasonal_order == (0, 0, 0, 0) and order[1] == 0:
            # điểm xuất phát = nghiệm OLS trên cùng biến (giúp hội tụ khi exog nhiều cột)
            beta, *_ = np.linalg.lstsq(exog, y, rcond=None)
            e = y - exog @ beta
            # AR bậc 1 lấy tự tương quan phần dư, các bậc cao hơn = 0 (lặp ρ cho mọi bậc ra điểm không dừng)
            arma = ([np.corrcoef(e[1:], e[:-1])[0, 1]] + [0.0] * (order[0] - 1) if order[0] else []) + [0.0] * order[2]
            start = np.r_[beta, arma, e.var()]
        res = mod.fit(start_params=start, disp=False, maxiter=MAXITER)
        if not res.mle_retvals.get("converged", False):
            powell = mod.fit(start_params=start, method="powell", disp=False, maxiter=5 * MAXITER)
            res = mod.fit(start_params=powell.params, disp=False, maxiter=MAXITER)
    return res


class _StateSpace(Forecaster):
    family = "statistical"
    order = (1, 0, 1)
    seasonal_order = (0, 0, 0, 0)

    def exog(self, dates):
        return None

    def fit(self, y, cal):
        self.cal, self.index = cal, y.index
        X = self.exog(y.index)
        self.res = _fit_sarimax(np.log(y.values), X, tuple(self.params.get("order", self.order)),
                                tuple(self.params.get("seasonal_order", self.seasonal_order)),
                                self.params.get("trend", "n"))
        self.converged = bool(self.res.mle_retvals.get("converged", False))
        return self

    def predict(self, dates):
        X = self.exog(dates)
        steps = len(dates)
        # Dự báo bắt đầu ngay sau train; dates phải liền sau origin
        if dates[0] != self.index[-1] + pd.Timedelta(days=1):
            raise ValueError("Ngày dự báo phải bắt đầu ngay sau origin")
        return np.exp(np.asarray(self.res.forecast(steps, exog=X)))

    def fitted(self):
        return pd.Series(np.exp(np.asarray(self.res.fittedvalues)), index=self.index)


def _fourier(cal, dates):
    X = cal.features(dates, "linear")
    return X[[c for c in X if c.startswith(("sin", "cos"))]].assign(const=1.0).values


class ARIMA(_StateSpace):
    name = "arima"
    GRID = [(p, d, q) for p, d, q in itertools.product((0, 1, 2), (0, 1), (0, 1, 2)) if p + q > 0]

    def fit(self, y, cal):
        """Chọn (p, d, q) bằng AIC trên train, d = 0 thì có hằng số."""
        self.cal, self.index = cal, y.index
        ly = np.log(y.values)
        best = None
        for order in self.params.get("grid", self.GRID):
            try:
                r = _fit_sarimax(ly, None, order, trend="c" if order[1] == 0 else "n")
            except Exception:
                continue
            if best is None or r.aic < best[0]:
                best = (r.aic, order, r)
        self.aic, self.selected_order, self.res = best
        self.converged = bool(self.res.mle_retvals.get("converged", False))
        return self


class SARIMAWeekly(_StateSpace):
    name = "sarima_w7"
    order, seasonal_order = (1, 1, 1), (1, 0, 1, 7)


class SARIMA(_StateSpace):
    name = "sarima"

    def exog(self, dates):
        return _fourier(self.cal, dates)


class SARIMAX(_StateSpace):
    name = "sarimax"
    calendar_aware = True
    drop_groups: tuple = ()

    def exog(self, dates):
        return self.cal.features(dates, "linear", drop=self.drop_groups).assign(const=1.0).values


class _Prophet(Forecaster):
    family = "statistical"

    def _frame(self, dates):
        return pd.DataFrame({"ds": pd.DatetimeIndex(dates)})

    def fit(self, y, cal):
        from prophet import Prophet
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)
        self.cal, self.index = cal, y.index
        self.m = Prophet(**self.prophet_kwargs())
        df = self._frame(y.index).assign(y=np.log(y.values))
        df = self.add_regressors(df)
        self.m.fit(df)
        return self

    def prophet_kwargs(self):
        return {}

    def add_regressors(self, df):
        return df

    def predict(self, dates):
        df = self.add_regressors(self._frame(dates))
        return np.exp(self.m.predict(df)["yhat"].values)

    def fitted(self):
        return pd.Series(self.predict(self.index), index=self.index)


class ProphetPure(_Prophet):
    name = "prophet_pure"


class ProphetCal(_Prophet):
    name = "prophet"
    calendar_aware = True

    def prophet_kwargs(self):
        return dict(growth=self.params.get("growth", "flat"), yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=self.params.get("changepoint_prior_scale", 0.05),
                    seasonality_prior_scale=self.params.get("seasonality_prior_scale", 10.0))

    def add_regressors(self, df):
        X = self.cal.features(df["ds"], "linear")
        X = X[[c for c in X if not c.startswith(("sin", "cos"))]]   # Prophet đã có mùa vụ năm riêng
        if not hasattr(self, "_reg_cols"):
            self._reg_cols = list(X.columns)
            for c in self._reg_cols:
                self.m.add_regressor(c, standardize=False)
        return pd.concat([df.reset_index(drop=True), X[self._reg_cols].reset_index(drop=True)], axis=1)
