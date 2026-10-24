"""Kiểm contract/metric/feature; không fit model và không cần dữ liệu thật."""
import os
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cloud_pilot import baseline, cloud_runtime, score, statistical_calendar, supervised, validate_data


class ProtocolTests(unittest.TestCase):
    def test_local_runtime_is_blocked(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                cloud_runtime()

    def test_metrics_hand_calculation(self):
        result = score([2, 4], [1, 4], np.arange(800))
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["rse"], 0.5)
        self.assertAlmostEqual(result["mape"], 25.0)
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.5))

    def test_undefined_metrics(self):
        result = score([0, 0], [1, 0], np.ones(800))
        for key in ("mape", "rse", "mase", "wape"):
            self.assertIsNone(result[key])
        self.assertEqual(result["mape_coverage"], 0.0)

    def test_placeholder_dates_rejected(self):
        dates = pd.date_range("2012-07-04", "2022-12-31")
        frame = pd.DataFrame({"sale_date": dates, "g": 1.0, "cogs": 2.0})
        self.assertEqual(len(validate_data(frame)), 3833)
        frame.loc[len(frame)] = [pd.Timestamp("2023-01-01"), 1.0, 1.0]
        with self.assertRaises(ValueError):
            validate_data(frame)

    def test_seasonal_calendar_leap_day_and_recursive(self):
        dates = pd.date_range("2017-01-01", "2019-12-31")
        train = pd.Series(np.arange(len(dates), dtype=float), index=dates)
        result = baseline(train, pd.date_range("2020-01-01", periods=548), "snaive_year")
        self.assertEqual(result[59], train.loc["2019-02-28"])
        self.assertEqual(result[366], train.loc["2019-01-01"])

    def test_features_do_not_read_future(self):
        config = {"lag_days": [1, 7], "rolling_days": [7]}
        series = pd.Series(np.arange(60, dtype=float), index=pd.date_range("2020-01-01", periods=60))
        train = series.loc[:"2020-01-31"]
        before = supervised(train, config)
        series.loc["2020-02-01":] = 99999
        after = supervised(series.loc[:"2020-01-31"], config)
        np.testing.assert_array_equal(before[0], after[0])
        self.assertEqual(before[0][0][0], 6.0)
        self.assertEqual(before[1][0], 7.0)

    def test_statistical_calendar_has_independent_known_future_columns(self):
        train_dates = pd.date_range("2012-07-04", "2019-12-31", freq="D")
        future_dates = pd.date_range("2020-01-01", periods=548, freq="D")
        train_exog = statistical_calendar(train_dates)
        future_exog = statistical_calendar(future_dates)
        self.assertEqual(train_exog.shape, (len(train_dates), 6))
        self.assertEqual(future_exog.shape, (548, 6))
        self.assertEqual(np.linalg.matrix_rank(train_exog), 6)
        self.assertTrue(np.isfinite(future_exog).all())
        np.testing.assert_array_equal(future_exog[0], statistical_calendar([future_dates[0]])[0])


if __name__ == "__main__":
    unittest.main()
