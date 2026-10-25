"""Kiểm giao thức: dữ liệu khớp, split không chồng test, không rò rỉ, metric đúng. Không fit model nặng."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from rsf import data, metrics, splits
from rsf.features import Calendar
from rsf.models import REGISTRY

HAS_DATA = (data.PROCESSED / data.SERIES_FILE).exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="chưa build data/processed")


@pytest.fixture(scope="module")
def loaded():
    return data.load()


@needs_data
def test_regions_sum_to_total(loaded):
    df, _ = loaded
    data.check(df)  # raise nếu lệch
    assert set(df.series_id) == set(data.SERIES)
    assert df.groupby("series_id").size().eq(3833).all()


def test_folds_never_touch_test():
    for f in splits.FOLDS:
        assert f.end <= splits.TEST_ORIGIN
        assert len(f.dates) == splits.VAL_DAYS
    assert splits.TEST.dates[0] == pd.Timestamp("2021-07-01")
    assert len(splits.TEST.dates) == 549


@needs_data
def test_promo_rules_only_use_past(loaded):
    _, promos = loaded
    for f in splits.FOLDS + [splits.TEST]:
        cal = Calendar(promos, f.origin)
        used = promos[promos.start_date <= f.origin]
        assert cal.rules["first_year"].max() <= f.origin.year
        assert len(cal.rules) == used["promo_name"].str.replace(r"\s+\d{4}$", "", regex=True).nunique()


@needs_data
def test_regime_flag_only_after_observed(loaded):
    _, promos = loaded
    assert "post2019" not in Calendar(promos, splits.FOLDS[1].origin).features(splits.FOLDS[1].dates).columns
    assert "post2019" in Calendar(promos, splits.TEST.origin).features(splits.TEST.dates).columns


@needs_data
def test_prediction_ignores_future_truth(loaded):
    """Làm hỏng y sau origin: dự báo của model nhanh không được đổi."""
    df, promos = loaded
    y = data.series(df, "TOTAL", "revenue")
    f = splits.FOLDS[3]
    broken = y.copy()
    broken.loc[f.origin + pd.Timedelta(days=1):] *= 10
    for name in ["snaive1y", "snaive2y", "lightgbm"]:
        p1 = REGISTRY[name](seed=42).fit(y.loc[:f.origin], Calendar(promos, f.origin)).predict(f.dates)
        p2 = REGISTRY[name](seed=42).fit(broken.loc[:f.origin], Calendar(promos, f.origin)).predict(f.dates)
        np.testing.assert_allclose(p1, p2)


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.uniform(1, 100, 50), rng.uniform(1, 100, 50)
    assert metrics.mae(y, p) == pytest.approx(mean_absolute_error(y, p))
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt(mean_squared_error(y, p)))
    assert metrics.mape(y, p) == pytest.approx(100 * mean_absolute_percentage_error(y, p))


def test_dm_sign():
    rng = np.random.default_rng(1)
    e = rng.normal(size=500)
    stat, p = metrics.diebold_mariano(0.5 * e, e)
    assert stat < 0 and p < 0.01
