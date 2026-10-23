"""Kiểm hợp đồng screening mà không fit model hoặc đọc dữ liệu thật."""
import json
from pathlib import Path
import unittest

from screening import FOLDS, NEURAL, STATISTICAL, SUPPORTED, TABULAR, validate_config

ROOT = Path(__file__).resolve().parents[1] / "configs"
REQUIRED = {
    "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost",
    "arima", "arimax", "sarima", "sarimax", "rnn", "lstm",
}


class ScreeningProtocolTests(unittest.TestCase):
    def config(self, kind, mode):
        return json.loads((ROOT / f"screening_{kind}_{mode}.json").read_text(encoding="utf-8"))

    def test_requested_models_registered(self):
        self.assertTrue(REQUIRED <= SUPPORTED)
        cpu = set(self.config("cpu", "pilot")["models"])
        gpu = set(self.config("gpu", "pilot")["models"])
        self.assertEqual(REQUIRED, (cpu | gpu) - {"snaive_year", "snaive_two_year"})
        self.assertEqual(cpu & TABULAR, TABULAR)
        self.assertEqual(cpu & STATISTICAL, STATISTICAL)
        self.assertEqual(gpu & NEURAL, NEURAL)

    def test_pilot_and_cv_keep_test_closed(self):
        for kind in ("cpu", "gpu"):
            pilot = self.config(kind, "pilot")
            cv = self.config(kind, "cv")
            validate_config(pilot)
            validate_config(cv)
            self.assertEqual(pilot["folds"], ["F5"])
            self.assertEqual(cv["folds"], list(FOLDS))
            self.assertEqual(pilot["horizons"], cv["horizons"])
            self.assertEqual(pilot["models"], cv["models"])
            self.assertEqual(pilot["targets"], ["G", "COGS"])

    def test_statistical_names_have_distinct_seasonality(self):
        options = self.config("cpu", "pilot")["statistical"]
        for name in ("sarima", "sarimax"):
            self.assertEqual(options[name]["seasonal_order"][-1], 7)
        for name in ("arima", "arimax"):
            self.assertNotIn("seasonal_order", options[name])

    def test_invalid_fold_and_gpu_on_cpu_rejected(self):
        config = self.config("cpu", "pilot")
        config["folds"] = ["final"]
        with self.assertRaises(ValueError):
            validate_config(config)
        config = self.config("cpu", "pilot")
        config["models"] = ["lstm"]
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()

