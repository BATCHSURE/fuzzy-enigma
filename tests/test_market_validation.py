"""Offline market-tool regressions. Run with the optional market dependencies.

Live LSEG is deliberately never invoked; captured fixtures and independent
QuantLib expected values are replayed without re-generating the oracle.
"""

from copy import deepcopy
import csv
from datetime import date
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# Also support `python tests/test_market_validation.py` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import QuantLib as ql
import fuzzy_enigma as fe

from tools.market_validation.analytics import (
    black_price, evaluation_date, implied_volatility, validate_snapshot,
)
from tools.market_validation.calibration import calibrate, heston_price, numerical_validation, representatives
from tools.market_validation.regression import make_fixture
from tools.market_validation.snapshot import copy_evidence, dataset_hash, read_curves, read_snapshot, redact, write_json

FIXTURES = ROOT / "tests/fixtures/market"


def snapshot():
    return read_snapshot(FIXTURES / "synthetic_spxw.json")


class SnapshotTests(unittest.TestCase):
    def test_roundtrip_hash_and_credentials(self):
        data = snapshot()
        data["diagnostic"] = {"app_key": "private-value", "nested": {"authorization": "Bearer token"},
                              "message": "app key " + "a" * 40}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.json"
            write_json(path, data)
            raw = path.read_text()
            self.assertNotIn("private-value", raw)
            self.assertNotIn("a" * 40, raw)
            self.assertEqual(read_snapshot(path)["spot"], data["spot"])
            restored = read_snapshot(path)
            self.assertEqual(dataset_hash(restored), dataset_hash(data))
            restored["versions"] = {"test": "1"}
            restored["dataset_sha256"] = "ignore-me"
            self.assertEqual(dataset_hash(restored), dataset_hash(data))

    def test_nonfinite_json_and_schema_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            with self.assertRaises(ValueError):
                write_json(path, {"price": float("nan")})
            path.write_text('{"schema_version":2}')
            with self.assertRaises(ValueError):
                read_snapshot(path)

    def test_replay_carries_and_verifies_raw_sidecar(self):
        import hashlib
        data = snapshot()
        with tempfile.TemporaryDirectory() as folder:
            origin = Path(folder)/"original"; origin.mkdir()
            raw = origin/"raw_responses.json"
            write_json(raw, {"data": [1, 2, 3]})
            data["source"].update(raw_response_file=raw.name,
                                  raw_response_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
            write_json(origin/"snapshot.json", data)
            destination = Path(folder)/"replay"
            copy_evidence(data, origin/"snapshot.json", destination)
            self.assertTrue((destination/"raw_responses.json").is_file())
            self.assertEqual(data["source"]["raw_response_status"], "verified")
            raw.write_text('{"data":[9]}')
            with self.assertRaises(ValueError):
                copy_evidence(data, origin/"snapshot.json", destination)
            data["source"]["raw_response_file"] = "../secret.json"
            with self.assertRaises(ValueError):
                copy_evidence(data, origin/"snapshot.json", destination)

    def test_explicit_curves_csv_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "curves.csv"
            row = snapshot()["curves"][0]
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader(); w.writerow(row)
            self.assertEqual(read_curves(path, row["as_of"]), [row])
            with self.assertRaises(ValueError):
                read_curves(path, "2026-10-05")
            with path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(row)).writerow(row)
            with self.assertRaises(ValueError):
                read_curves(path, row["as_of"])


class AnalyticsTests(unittest.TestCase):
    def test_normalization_and_grouped_holdout(self):
        result = validate_snapshot(snapshot())
        self.assertGreaterEqual(len(result["accepted"]), 40)
        splits = {}
        for row in result["accepted"]:
            self.assertAlmostEqual(row["r"], .03, places=12)
            self.assertAlmostEqual(row["q"], .012, places=12)
            self.assertAlmostEqual(row["t"], (date.fromisoformat(row["expiry"]) - date(2026, 10, 6)).days / 365)
            self.assertAlmostEqual(black_price(row, row["iv"]), row["close"], places=8)
            self.assertEqual(splits.setdefault(row["group"], row["split"]), row["split"])
            self.assertEqual(row["reference_kind"], "close")
        self.assertIn("holdout", splits.values())
        reps = representatives(result["accepted"])
        self.assertEqual(len(reps), len({row["group"] for row in reps}))
        self.assertTrue(all(row["kind"] == ("call" if row["strike"] >= row["forward"] else "put") for row in reps))

    def test_bad_date_units_terms_price_and_duplicates(self):
        data = snapshot()
        data["quotes"] = [deepcopy(data["quotes"][8]) for _ in range(7)]
        for i, row in enumerate(data["quotes"]):
            row["ric"] = f"bad-{i}"
        data["quotes"][0]["close_date"] = "2026-10-05"
        data["quotes"][1]["premium_unit"] = "contract_dollars"
        data["quotes"][2]["exercise_style"] = "American"
        data["quotes"][3]["settlement"] = "AM"
        data["quotes"][4]["close"] = 2000.
        data["quotes"][6]["ric"] = data["quotes"][5]["ric"]
        result = validate_snapshot(data)
        self.assertEqual(len(result["accepted"]), 1)
        self.assertEqual(len(result["rejected"]), 6)

    def test_missing_or_wrong_date_curves_reject_quotes(self):
        for curves in ([], [{**curve, "as_of": "2026-10-05"} for curve in snapshot()["curves"]]):
            data = snapshot(); data["curves"] = curves
            self.assertEqual(validate_snapshot(data)["accepted"], [])

    def test_duplicate_curve_cannot_silently_choose_one(self):
        data = snapshot()
        expiry = data["curves"][0]["expiry"]
        data["curves"].append({**data["curves"][0], "forward": 200.0})
        result = validate_snapshot(data)
        self.assertFalse(any(row["expiry"] == expiry for row in result["accepted"]))

    def test_historical_spreads_only_and_frozen_weights(self):
        data = snapshot(); row = data["quotes"][8]
        row.update(bid=row["close"] - .03, ask=row["close"] + .03,
                   bid_date="2026-10-06", ask_date="2026-10-06")
        validated = validate_snapshot(data)
        observed = next(item for item in validated["accepted"] if item["ric"] == row["ric"])
        self.assertTrue(observed["bid_ask_usable"])
        self.assertAlmostEqual(observed["uncertainty_price"], .05)
        row["bid_date"] = "2026-10-07"
        stale = next(item for item in validate_snapshot(data)["accepted"] if item["ric"] == row["ric"])
        self.assertFalse(stale["bid_ask_usable"])
        self.assertEqual(stale["uncertainty_source"], "assumed_close_iv_uncertainty")

    def test_iv_bounds_no_price_clipping_and_ql_date_restored(self):
        row = validate_snapshot(snapshot())["accepted"][0]
        with self.assertRaises(ValueError):
            implied_volatility(row, -1)
        with self.assertRaises(ValueError):
            implied_volatility(row, 10000)
        previous = ql.Settings.instance().evaluationDate
        with self.assertRaises(RuntimeError):
            with evaluation_date("2026-10-06"):
                raise RuntimeError("probe")
        self.assertEqual(ql.Settings.instance().evaluationDate, previous)

    def test_vendor_surface_requires_explicit_matching_conventions(self):
        data = snapshot()
        point = {"expiry": data["quotes"][0]["expiry"], "strike": 100., "iv": .2}
        data["vendor_surface"] = {"as_of": "2026-10-05", "volatility_unit": "decimal", "convention": "black", "points": [point]}
        self.assertIsNone(validate_snapshot(data)["vendor_surface"])
        data["vendor_surface"]["as_of"] = "2026-10-06"
        self.assertEqual(validate_snapshot(data)["vendor_surface"]["points"], [point])


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = validate_snapshot(snapshot())
        cls.fit = calibrate(cls.data, starts=1, max_nfev=100, seed=42)

    def test_fit_and_heldout_repricing(self):
        self.assertEqual(self.fit["status"], "converged")
        self.assertLess(self.fit["train"]["price_rmse"], 1e-5)
        self.assertLess(self.fit["holdout"]["price_rmse"], 1e-5)
        self.assertEqual(set(self.fit["parameters"]), {"v0", "kappa", "theta", "xi", "rho"})

    def test_holdout_and_vendor_do_not_select_parameters(self):
        altered = deepcopy(self.data)
        for row in altered["accepted"]:
            if row["split"] == "holdout":
                row["close"] *= 1.20
        altered["vendor_surface"] = {"points": [{"iv": 7.0}]}
        fit = calibrate(altered, starts=1, max_nfev=100, seed=42)
        self.assertEqual(fit["parameters"], self.fit["parameters"])
        self.assertEqual(fit["chosen_start"], self.fit["chosen_start"])

    def test_insufficient_training_and_bad_parameters(self):
        reduced = deepcopy(self.data)
        expiry = reduced["accepted"][0]["expiry"]
        reduced["accepted"] = [row for row in reduced["accepted"] if row["expiry"] == expiry]
        self.assertEqual(calibrate(reduced)["status"], "insufficient_data")
        params = snapshot()["synthetic_parameters"]
        with self.assertRaises(ValueError):
            heston_price(self.data["accepted"][0], {**params, "rho": 2.})

    def test_sparse_expiry_is_reported_and_does_not_affect_fit(self):
        data = deepcopy(self.data)
        sparse = {**data["accepted"][0], "ric": "sparse-expiry", "expiry": "2027-09-30",
                  "group": "2027-09-30:90", "split": "train", "kind": "put", "strike": 90.,
                  "t": 359/365, "discount_factor": math.exp(-.03*359/365),
                  "forward": 100*math.exp(.018*359/365)}
        sparse["close"] = black_price(sparse, .2)
        data["accepted"].append(sparse)
        fit = calibrate(data, starts=1, max_nfev=100, seed=42)
        self.assertEqual(fit["parameters"], self.fit["parameters"])
        self.assertEqual(fit["excluded_expiries"][0]["expiry"], "2027-09-30")
        prediction = next(row for row in fit["predictions"] if row["ric"] == sparse["ric"])
        self.assertEqual(prediction["calibration_role"], "excluded_expiry")
        self.assertEqual(prediction["split"], "excluded_expiry")

    def test_deterministic_variance_limit_and_small_mc_check(self):
        row = self.data["accepted"][8]
        params = dict(v0=.04, kappa=1.8, theta=.04, xi=0., rho=-.7)
        self.assertAlmostEqual(heston_price(row, params), black_price(row, .2), places=11)
        result = numerical_validation(self.data, paths=4000, steps=(1, 16), seeds=(42,))
        self.assertTrue(result["records"])
        self.assertTrue(all(math.isfinite(item["mc_price"]) for item in result["records"]))

    def test_seed_validation_preserves_independent_standard_errors(self):
        empty = {"accepted": [], "source": {}}
        for seeds in ((42, 42), (-1,), (2**64,), (42.0,), (True,), ("42",), (), None):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                numerical_validation(empty, seeds=seeds)
        result = numerical_validation(empty, seeds=(0, 2**64-1))
        self.assertEqual(result["settings"]["seeds"], [0, 2**64-1])


class LsegAdapterTests(unittest.TestCase):
    def test_search_multivalue_underlying_and_metadata(self):
        from tools.market_validation.lseg import MarketDataError, _verify_contract
        row = {"RIC": "SPXW-test", "OptionStub": "SPXW.U", "UnderlyingQuoteRIC": [".SPX"],
               "Currency": "USD", "CallPutOption": "Call", "StrikePrice": 7800,
               "ExpiryDate": "2026-11-05T00:00:00"}
        result = _verify_contract(row)
        self.assertEqual(result["expiry"], "2026-11-05")
        self.assertEqual(result["expiry_source_value"], row["ExpiryDate"])
        self.assertIsNone(result["expiry_utc"])
        with self.assertRaises(MarketDataError):
            _verify_contract({**row, "UnderlyingQuoteRIC": [".SPX", "SPY.P"]})

    def test_native_history_preserves_close_date_and_bid_ask(self):
        from types import SimpleNamespace
        from tools.market_validation.lseg import _native_option_closes
        ric = "SPXW-test"
        raw = {"universe": {"ric": ric}, "qos": {"timeliness": "delayed"},
               "headers": [{"name": name} for name in ("DATE", "TRDPRC_1", "BID", "ASK")],
               "data": [["2026-10-06", 94.77, 93.4, 98.1]]}
        response = SimpleNamespace(is_success=True, data=SimpleNamespace(raw=raw))
        with patch("lseg.data.content.historical_pricing.summaries.Definition") as definition:
            definition.return_value.get_data.return_value = response
            result, audit = _native_option_closes(object(), [ric], date(2026, 10, 6))
            self.assertEqual(result[ric]["close"], 94.77)
            self.assertEqual(result[ric]["bid_date"], "2026-10-06")
            self.assertEqual(result[ric]["delay_status"], "delayed")
            raw["data"][0][0] = "2026-10-05"
            result, audit = _native_option_closes(object(), [ric], date(2026, 10, 6))
            self.assertEqual(result, {})

    def test_forward_curve_temporal_gate_and_bounded_interpolation(self):
        from tools.market_validation.lseg import _ipa_curves
        as_of = "2026-10-06"; expiry = "2026-11-05"
        raw = {"request_context": {"as_of": as_of, "price_basis": "close"}, "data": [{
            "underlyingSpot": [{"instrumentCode": ".SPX", "priceDate": as_of, "price": 7818.93}],
            "discountCurve": {"curveParameters": {"marketDataDate": as_of}, "points": [
                {"endDate": as_of, "discountFactor": 1.},
                {"endDate": "2026-12-05", "discountFactor": .99}]},
            "forwardCurve": {"dataPoints": {expiry: 7845.979}},
        }]}
        curves = _ipa_curves(raw, date.fromisoformat(as_of), [expiry])
        self.assertEqual(len(curves), 1)
        self.assertEqual(curves[0]["forward"], 7845.979)
        self.assertAlmostEqual(curves[0]["discount_factor"], math.sqrt(.99))
        self.assertEqual(_ipa_curves(raw, date.fromisoformat(as_of), ["2027-01-05"]), [])
        raw["data"][0]["underlyingSpot"][0]["priceDate"] = "2026-10-05"
        self.assertEqual(_ipa_curves(raw, date.fromisoformat(as_of), [expiry]), [])

    def test_doctor_reports_missing_key_without_credential_leak(self):
        from tools.market_validation.lseg import MarketDataError, doctor
        with patch("tools.market_validation.lseg._desktop", side_effect=MarketDataError("app_key_missing", "Set LSEG_APP_KEY")):
            result = doctor()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["checks"]["session"]["code"], "app_key_missing")


class FrozenReferenceTests(unittest.TestCase):
    def test_every_committed_reference_case(self):
        paths = sorted(FIXTURES.glob("*_reference.json"))
        self.assertTrue(paths, "Need at least one frozen independent reference fixture")
        for path in paths:
            data = json.loads(path.read_text())
            self.assertEqual(data["reference"]["library"], "QuantLib")
            for row in data["cases"]:
                with self.subTest(fixture=path.name, ric=row["ric"]):
                    model = fe.GbmModel(row["spot"], row["r"], row["q"], row["sigma"], row["t"], steps=1)
                    self.assertAlmostEqual(fe.black_scholes_price(model, row["kind"], row["strike"]), row["expected_black"], delta=row["analytic_tolerance"])
                    engine = fe.McEngine(paths=row["mc"]["paths"], seed=row["mc"]["seed"])
                    result = engine.price_european(model, row["kind"], row["strike"])
                    self.assertLessEqual(abs(result.price-row["expected_black"]), 5*result.std_error+row["analytic_tolerance"])

    def test_new_analytic_api_domain_checks(self):
        model = fe.GbmModel(100., .03, .01, .2, 1.)
        for strike in (-1., 0., math.nan, math.inf):
            with self.assertRaises(ValueError):
                fe.black_scholes_price(model, "call", strike)
        with self.assertRaises(ValueError):
            fe.black_scholes_price(model, "invalid", 100.)
        heston = fe.HestonModel(100., .03, .01, .04, 1., .04, .2, -.7, 1.)
        with self.assertRaises(ValueError):
            fe.black_scholes_price(heston, "call", 100.)

    def test_frozen_heston_reference_and_core_atm_prices(self):
        cases = []
        for path in sorted(FIXTURES.glob("*_reference.json")):
            for row in json.loads(path.read_text())["cases"]:
                if row.get("heston_parameters"):
                    self.assertAlmostEqual(heston_price(row, row["heston_parameters"]),
                                           row["expected_heston"], delta=row["analytic_tolerance"])
                    cases.append(row)
        self.assertTrue(cases)
        by_expiry = {}
        for row in cases:
            if row["expected_heston"] > row["spot"] * 1e-4:
                old = by_expiry.get(row["expiry"])
                if old is None or abs(math.log(row["strike"]/row["forward"])) < abs(math.log(old["strike"]/old["forward"])):
                    by_expiry[row["expiry"]] = row
        self.assertTrue(by_expiry)
        for row in by_expiry.values():
            params = row["heston_parameters"]
            model = fe.HestonModel(row["spot"], row["r"], row["q"], params["v0"], params["kappa"],
                                   params["theta"], params["xi"], params["rho"], row["t"], steps=1024)
            result = fe.McEngine(paths=120_000, seed=42).price_european(model, row["kind"], row["strike"])
            self.assertLessEqual(abs(result.price-row["expected_heston"]), 5*result.std_error+row["analytic_tolerance"])

    def test_fixture_generation_preserves_data_origin(self):
        fixture = make_fixture(snapshot())
        self.assertEqual(fixture["source"]["provider"], "Synthetic")
        self.assertGreaterEqual(len(fixture["cases"]), 10)


class CLITests(unittest.TestCase):
    def test_offline_report_cannot_contact_lseg(self):
        from tools.market_validation.__main__ import main
        with tempfile.TemporaryDirectory() as folder, patch("tools.market_validation.lseg._desktop", side_effect=AssertionError("network attempted")):
            result = main(["report", "--snapshot", str(FIXTURES / "synthetic_spxw.json"),
                           "--output", folder, "--no-calibration", "--no-mc"])
            self.assertEqual(result, 0)
            self.assertTrue((Path(folder)/"report.html").is_file())
            self.assertTrue((Path(folder)/"validated.json").is_file())

    def test_cli_help_uses_only_stdlib(self):
        result = subprocess.run([sys.executable, "-S", "-m", "tools.market_validation", "--help"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("calibrate", result.stdout)


if __name__ == "__main__":
    unittest.main()
