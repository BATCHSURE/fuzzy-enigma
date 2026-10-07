"""Rare-event validation regressions using frozen, dated LSEG contracts."""

import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.market_validation.calibration import numerical_validation
from tools.market_validation.report import _write_csv


class TailValidationTests(unittest.TestCase):
    def test_both_real_tail_contracts_are_rechecked_with_nonzero_precision(self):
        fixture = json.loads((ROOT / "tests/fixtures/market/lseg_spxw_20261006_reference.json").read_text())
        rows = [deepcopy(row) for row in fixture["cases"]
                if row["ric"] in {"SPXWv132663000.U", "SPXWj132686000.U"}]
        self.assertEqual(len(rows), 2)
        for row in rows:
            row.update(group=f'{row["expiry"]}:{row["strike"]}', split="train")
        result = numerical_validation({"source": fixture["source"], "accepted": rows},
                                      params=rows[0]["heston_parameters"], paths=100_000,
                                      steps=(256, 1024), seeds=(42, 43))
        heston = [row for row in result["summaries"] if row["model"] == "Heston"]
        self.assertEqual(len(heston), 2)
        for row in heston:
            with self.subTest(ric=row["ric"]):
                self.assertEqual(row["estimator"], "conditional_importance")
                self.assertTrue(row["tail_resolution"])
                self.assertGreater(row["mean_price"], 0)
                self.assertGreater(row["combined_std_error"], 0)
                self.assertLess(row["combined_std_error"] / row["mean_price"], .15)
                self.assertLessEqual(abs(row["price_error"]), 4 * row["combined_std_error"] + 1e-10)
                self.assertEqual(row["classification"], "within_sampling_error")
        self.assertTrue(any(row["estimator"] == "plain" for row in result["records"]))
        self.assertTrue(any(row["estimator"] == "conditional_importance" for row in result["records"]))

    def test_csv_preserves_mixture_and_reference_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tail.csv"
            _write_csv(path, [{"importance_shift": [0., -4., 4.],
                               "reference_convergence": {"error": 3e-9}}],
                       ["importance_shift", "reference_convergence"])
            with path.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(json.loads(row["importance_shift"]), [0., -4., 4.])
            self.assertEqual(json.loads(row["reference_convergence"]), {"error": 3e-9})


if __name__ == "__main__":
    unittest.main()
