"""Independent, offline SSVI tests and explicitly invoked QuantLib fixtures.

The frozen oracle is produced by QuantLib's raw-SVI smile section, using
its documented parameter order ``a, b, sigma, rho, m``. Ordinary tests only
read it. To deliberately refresh the synthetic oracle, run this file with
``--generate-fixtures``; no project surface implementation is imported by
that generator.
"""

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, timedelta
import io
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURES = ROOT / "tests/fixtures/market"


def generate_fixtures():
    """Explicit developer operation: write synthetic QuantLib-only oracles."""
    import QuantLib as ql

    as_of = date(2026, 10, 6)
    spot, r, q, rho, eta = 100.0, 0.03, 0.01, -0.55, 0.85
    day_nodes = [(30, 0.005), (90, 0.013), (180, 0.026), (360, 0.052)]
    nodes = [{"expiry": (as_of + timedelta(days=days)).isoformat(),
              "t": days / 365.0, "theta": theta} for days, theta in day_nodes]
    curves = [{"as_of": as_of.isoformat(), "expiry": node["expiry"],
               "discount_factor": math.exp(-r * node["t"]),
               "forward": spot * math.exp((r - q) * node["t"]),
               "source": "Synthetic explicit flat r=0.03, q=0.01"} for node in nodes]

    def theta_for(days):
        for index in range(1, len(day_nodes)):
            left, right = day_nodes[index - 1], day_nodes[index]
            if left[0] <= days <= right[0]:
                weight = (days - left[0]) / (right[0] - left[0])
                return left[1] * (1.0 - weight) + right[1] * weight
        raise ValueError("oracle requested outside its node interval")

    def section(days):
        t, theta = days / 365.0, theta_for(days)
        forward = spot * math.exp((r - q) * t)
        # Raw-SVI equivalence to gamma=1/2 modified-power SSVI. QuantLib
        # evaluates variance and option prices; project SSVI is not called.
        phi = eta / math.sqrt(theta * (1.0 + theta))
        parameters = [0.5 * theta * (1.0 - rho * rho), 0.5 * theta * phi,
                      math.sqrt(1.0 - rho * rho) / phi, rho, -rho / phi]
        return ql.SviSmileSection(t, forward, parameters), forward

    quotes, coverage = [], []
    for days, _theta in day_nodes:
        smile, forward = section(days)
        expiry = (as_of + timedelta(days=days)).isoformat()
        discount = math.exp(-r * days / 365.0)
        for index in range(9):
            k = -0.18 + 0.045 * index
            strike = forward * math.exp(k)
            coverage.append({"k": k, "t": days / 365.0})
            for kind, option in (("call", ql.Option.Call), ("put", ql.Option.Put)):
                premium = float(smile.optionPrice(strike, option, discount))
                quotes.append({"ric": f"SYN-SSVI-{expiry}-{index}-{kind}", "expiry": expiry,
                               "strike": strike, "kind": kind, "close": premium,
                               "close_date": as_of.isoformat(), "exercise_style": "European",
                               "settlement": "PM", "currency": "USD", "premium_unit": "index_points",
                               "bid": None, "ask": None, "vendor_iv": None,
                               "expiry_utc": expiry + "T20:00:00Z",
                               "terms_source": "Synthetic European PM test contract"})
    generator = {"library": "QuantLib", "version": ql.__version__, "engine": "SviSmileSection",
                 "svi_parameter_order": ["a", "b", "sigma", "rho", "m"],
                 "mapping_source": "https://github.com/lballabio/QuantLib/blob/master/ql/experimental/volatility/sviinterpolation.hpp",
                 "ssvi_phi": "eta/sqrt(theta*(1+theta)); gamma=1/2",
                 "regeneration": "python tests/test_market_vol_surface.py --generate-fixtures"}
    snapshot = {"schema_version": 1,
                "source": {"provider": "Synthetic", "as_of": as_of.isoformat(), "price_basis": "close",
                           "description": "Synthetic European quotes from independent QuantLib raw-SVI; no LSEG observations."},
                "spot": spot, "curves": curves, "quotes": quotes, "generator": generator,
                "synthetic_parameters": {"rho": rho, "eta": eta, "theta_nodes": nodes}}
    cases = []
    for days in (30, 60, 90, 120, 180, 270, 360):
        smile, forward = section(days)
        expiry, t = (as_of + timedelta(days=days)).isoformat(), days / 365.0
        discount = math.exp(-r * t)
        for k in (-3.0, -1.0, -0.14, -0.07, 0.0, 0.07, 0.14, 1.0, 3.0):
            strike = forward * math.exp(k)
            cases.append({"expiry": expiry, "t": t, "k": k, "strike": strike,
                          "forward": forward, "discount_factor": discount,
                          "allow_wing": abs(k) > 0.18,
                          "expected_total_variance": float(smile.variance(strike)),
                          "expected_iv": float(smile.volatility(strike)),
                          "expected_call_price": float(smile.optionPrice(strike, ql.Option.Call, discount)),
                          "expected_put_price": float(smile.optionPrice(strike, ql.Option.Put, discount))})
    reference = {"schema_version": 1, "provider": "Synthetic", "as_of": as_of.isoformat(),
                 "rho": rho, "eta": eta, "spot": spot, "r": r, "q": q,
                 "nodes": nodes, "curves": curves, "coverage": coverage,
                 "generator": generator, "cases": cases}
    for filename, value in (("synthetic_ssvi_quotes.json", snapshot),
                            ("synthetic_ssvi_oracle.json", reference)):
        (FIXTURES / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__" and "--generate-fixtures" in sys.argv:
    generate_fixtures()
    raise SystemExit(0)

from tools.market_validation.analytics import validate_snapshot
from tools.market_validation.surface import VolSurface, fit_ssvi
from tools.market_validation.snapshot import dataset_hash


def reference():
    return json.loads((FIXTURES / "synthetic_ssvi_oracle.json").read_text())


def quotes():
    return json.loads((FIXTURES / "synthetic_ssvi_quotes.json").read_text())


def oracle_surface(**changes):
    data = reference()
    arguments = {key: deepcopy(data[key]) for key in ("as_of", "rho", "eta", "nodes", "curves", "coverage")}
    arguments.update(changes)
    return VolSurface(**arguments)


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def refresh_outer_hash(value):
    value["surface_sha256"] = canonical_hash({key: item for key, item in value.items() if key != "surface_sha256"})


class QuantLibSurfaceOracleTests(unittest.TestCase):
    def test_frozen_raw_svi_mapping_variance_volatility_and_black_price(self):
        surface = oracle_surface()
        data = reference()
        self.assertEqual(data["provider"], "Synthetic")
        self.assertEqual(data["generator"]["svi_parameter_order"], ["a", "b", "sigma", "rho", "m"])
        for case in data["cases"]:
            with self.subTest(expiry=case["expiry"], k=case["k"]):
                wing = case["allow_wing"]
                self.assertAlmostEqual(surface.total_variance(case["k"], case["t"], allow_wing=wing),
                                       case["expected_total_variance"], delta=2e-12)
                self.assertAlmostEqual(surface.implied_vol(case["strike"], case["expiry"], allow_wing=wing),
                                       case["expected_iv"], delta=2e-12)
                for kind in ("call", "put"):
                    self.assertAlmostEqual(surface.black_price(kind, case["strike"], case["expiry"], allow_wing=wing),
                                           case[f"expected_{kind}_price"], delta=2e-10)

    def test_eta_zero_has_black_oracle_and_no_strike_skew(self):
        import QuantLib as ql
        surface = oracle_surface(eta=0.0)
        for case in reference()["cases"]:
            with self.subTest(expiry=case["expiry"], k=case["k"]):
                theta = surface.total_variance(0.0, case["t"])
                self.assertAlmostEqual(surface.total_variance(case["k"], case["t"], allow_wing=True), theta, places=14)
                derivatives = surface.derivatives(case["k"], case["t"], allow_wing=True)
                self.assertEqual(derivatives["w_k"], 0.0)
                self.assertEqual(derivatives["w_kk"], 0.0)
                for kind, option in (("call", ql.Option.Call), ("put", ql.Option.Put)):
                    expected = ql.blackFormula(option, case["strike"], case["forward"], math.sqrt(theta), case["discount_factor"])
                    self.assertAlmostEqual(surface.black_price(kind, case["strike"], case["expiry"], allow_wing=True),
                                           expected, delta=2e-10)

    def test_analytic_derivatives_match_independent_finite_differences(self):
        surface = oracle_surface()
        for days in (45, 120, 270):
            t = days / 365.0
            for k in (-2.0, -0.1, 0.0, 0.1, 2.0):
                with self.subTest(days=days, k=k):
                    derivative = surface.derivatives(k, t, allow_wing=True)
                    h = 1e-5
                    mid = surface.total_variance(k, t, allow_wing=True)
                    up = surface.total_variance(k + h, t, allow_wing=True)
                    down = surface.total_variance(k - h, t, allow_wing=True)
                    self.assertAlmostEqual(derivative["w"], mid, places=14)
                    first_fd = (up - down) / (2 * h)
                    self.assertTrue(math.isclose(derivative["w_k"], first_fd, rel_tol=1e-6, abs_tol=1e-11),
                                    (derivative["w_k"], first_fd))
                    # A larger second-difference spacing controls floating
                    # cancellation in the well-conditioned wing points.
                    h_second = 5e-4 * (1 + abs(k))
                    second_fd = (surface.total_variance(k + h_second, t, allow_wing=True) - 2*mid +
                                 surface.total_variance(k - h_second, t, allow_wing=True)) / h_second**2
                    self.assertTrue(math.isclose(derivative["w_kk"], second_fd, rel_tol=1e-4, abs_tol=1e-10),
                                    (derivative["w_kk"], second_fd))
                    ht = 1e-5
                    time_fd = (surface.total_variance(k, t + ht, allow_wing=True) -
                               surface.total_variance(k, t - ht, allow_wing=True)) / (2 * ht)
                    self.assertTrue(math.isclose(derivative["w_t"], time_fd, rel_tol=1e-6, abs_tol=1e-11),
                                    (derivative["w_t"], time_fd))

    def test_calendar_density_and_call_convexity_include_far_wings(self):
        surface = oracle_surface()
        nodes = reference()["nodes"]
        times = [nodes[0]["t"] + (nodes[-1]["t"] - nodes[0]["t"]) * index / 30 for index in range(31)]
        for k in (-20.0, -3.0, -0.18, 0.0, 0.18, 3.0, 20.0):
            values = [surface.total_variance(k, t, allow_wing=True) for t in times]
            self.assertTrue(all(right >= left - 1e-13 for left, right in zip(values, values[1:])))
            for t in times:
                d = surface.derivatives(k, t, allow_wing=True)
                w, first, second = d["w"], d["w_k"], d["w_kk"]
                density_factor = (1 - k * first / (2 * w))**2 - 0.25 * first**2 * (1/w + 0.25) + 0.5 * second
                self.assertGreaterEqual(density_factor, -1e-12)
        for curve in reference()["curves"]:
            forward, expiry, discount = curve["forward"], curve["expiry"], curve["discount_factor"]
            strikes = [forward * (0.05 + 0.02 * index) for index in range(146)]
            prices = [surface.black_price("call", strike, expiry, allow_wing=True) for strike in strikes]
            slopes = [(right - left) / (k2 - k1) for left, right, k1, k2 in zip(prices, prices[1:], strikes, strikes[1:])]
            self.assertTrue(all(-discount - 1e-10 <= slope <= 1e-10 for slope in slopes))
            self.assertTrue(all(right >= left - 1e-9 for left, right in zip(slopes, slopes[1:])))
            for strike, price in zip(strikes, prices):
                self.assertGreaterEqual(price, discount * max(forward - strike, 0) - 1e-10)
                self.assertLessEqual(price, discount * forward + 1e-10)

    def test_node_time_derivatives_choose_an_explicit_one_sided_slope(self):
        surface = oracle_surface()
        # ATM w is exactly theta; the unequal adjacent theta slopes give
        # independently known one-sided derivatives at the 90-day knot.
        nodes = reference()["nodes"]
        derivative = surface.derivatives(0.0, nodes[1]["t"])
        left = (nodes[1]["theta"] - nodes[0]["theta"]) / (nodes[1]["t"] - nodes[0]["t"])
        right = (nodes[2]["theta"] - nodes[1]["theta"]) / (nodes[2]["t"] - nodes[1]["t"])
        self.assertAlmostEqual(derivative["w_t_left"], left, places=12)
        self.assertAlmostEqual(derivative["w_t_right"], right, places=12)
        self.assertAlmostEqual(derivative["w_t"], right, places=12)
        self.assertEqual(derivative["time_side"], "right")
        chosen_left = surface.derivatives(0.0, nodes[1]["t"], time_side="left")
        self.assertAlmostEqual(chosen_left["w_t"], left, places=12)
        self.assertEqual(chosen_left["time_side"], "left")
        final = surface.derivatives(0.0, nodes[-1]["t"])
        self.assertEqual(final["time_side"], "left")
        self.assertIsNone(final["w_t_right"])
        self.assertEqual(final["w_t"], final["w_t_left"])
        with self.assertRaises(ValueError):
            surface.derivatives(0.0, nodes[1]["t"], time_side="central")


class SurfaceDomainAndPersistenceTests(unittest.TestCase):
    def test_default_coverage_and_time_extrapolation_are_rejected(self):
        surface = oracle_surface()
        t = 120 / 365.0
        with self.assertRaises(ValueError):
            surface.total_variance(0.4, t)
        self.assertGreater(surface.total_variance(0.4, t, allow_wing=True), 0.0)
        self.assertTrue(surface.derivatives(0.4, t, allow_wing=True)["wing_extrapolation"])
        self.assertFalse(surface.derivatives(0.0, t)["wing_extrapolation"])
        for outside in (0.0, 7/365.0, 400/365.0):
            for wing in (False, True):
                with self.subTest(t=outside, allow_wing=wing), self.assertRaises(ValueError):
                    surface.total_variance(0.0, outside, allow_wing=wing)
        for expiry in ("2026-10-06", "2026-10-13", "2027-11-10"):
            with self.subTest(expiry=expiry), self.assertRaises(ValueError):
                surface.implied_vol(100.0, expiry, allow_wing=True)

    def test_convex_hull_rejects_unobserved_corner_within_bounding_box(self):
        data = reference()
        first, last = data["nodes"][0]["t"], data["nodes"][-1]["t"]
        coverage = [{"k": -0.2, "t": first}, {"k": 0.2, "t": first}, {"k": 0.0, "t": last}]
        surface = oracle_surface(coverage=coverage)
        near_last = first + 0.9 * (last - first)
        with self.assertRaises(ValueError):
            surface.total_variance(0.15, near_last)
        self.assertGreater(surface.total_variance(0.15, near_last, allow_wing=True), 0.0)

    def test_constructor_rejects_dates_carry_and_constraint_tampering(self):
        data = reference()
        variants = [dict(rho=1.0), dict(rho=-1.0), dict(rho=float("nan")), dict(rho=True),
                    dict(eta=-0.1), dict(eta=float("inf")), dict(eta=3.0),
                    dict(epsilon=-1.0), dict(epsilon=float("nan"))]
        invalid_nodes = deepcopy(data["nodes"]); invalid_nodes[1]["theta"] = 0.001
        variants.append(dict(nodes=invalid_nodes))
        invalid_nodes = deepcopy(data["nodes"]); invalid_nodes[1]["t"] += 0.001
        variants.append(dict(nodes=invalid_nodes))
        invalid_nodes = deepcopy(data["nodes"]); invalid_nodes[1]["theta"] = float("nan")
        variants.append(dict(nodes=invalid_nodes))
        variants.append(dict(nodes=data["nodes"] + [deepcopy(data["nodes"][-1])]))
        variants.append(dict(nodes=list(reversed(data["nodes"]))))
        for field, value in (("as_of", "2026-10-05"), ("discount_factor", 0.0),
                             ("forward", -1.0), ("forward", float("inf"))):
            curves = deepcopy(data["curves"]); curves[0][field] = value
            variants.append(dict(curves=curves))
        curves = deepcopy(data["curves"]); curves.append(deepcopy(curves[0]))
        variants.append(dict(curves=curves))
        for changes in variants:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                oracle_surface(**changes)

    def test_query_domains_and_allow_wing_never_extrapolate_carry(self):
        surface = oracle_surface()
        for strike in (-1.0, 0.0, float("nan"), float("inf")):
            with self.subTest(strike=strike), self.assertRaises(ValueError):
                surface.implied_vol(strike, "2027-01-04")
        for kind in ("bad", "CALL", None):
            with self.subTest(kind=kind), self.assertRaises((ValueError, TypeError)):
                surface.black_price(kind, 100.0, "2027-01-04")
        for k, t in ((float("nan"), 0.3), (float("inf"), 0.3), (0., float("nan")), (0., float("inf"))):
            with self.subTest(k=k, t=t), self.assertRaises(ValueError):
                surface.total_variance(k, t, allow_wing=True)
        # Narrower carry dates cannot silently be extended to variance dates.
        data = reference()
        try:
            limited = oracle_surface(curves=data["curves"][1:-1])
        except ValueError:
            return  # Strict construction is also a valid rejection.
        with self.assertRaises(ValueError):
            limited.black_price("call", 100., data["nodes"][0]["expiry"], allow_wing=True)

    def test_serialization_roundtrip_validates_untrusted_parameter_files(self):
        surface = oracle_surface()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "surface.json"
            surface.save(path)
            restored = VolSurface.load(path)
            self.assertEqual(restored.to_dict(), surface.to_dict())
            case = reference()["cases"][20]
            self.assertAlmostEqual(restored.total_variance(case["k"], case["t"], allow_wing=True),
                                   case["expected_total_variance"], places=12)
        for field, value in (("model", "Heston"), ("schema_version", 2),
                             ("parameterisation", "unrestricted_svi"), ("rho", 1.01),
                             ("eta", 5.0), ("eta", float("nan"))):
            data = deepcopy(surface.to_dict()); data[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                VolSurface.from_dict(data)

    def test_mutating_input_or_public_copies_cannot_bypass_validation(self):
        data = reference()
        nodes, curves, coverage = deepcopy(data["nodes"]), deepcopy(data["curves"]), deepcopy(data["coverage"])
        surface = VolSurface(data["as_of"], data["rho"], data["eta"], nodes, curves, coverage)
        expected = surface.total_variance(0.0, nodes[1]["t"])
        nodes[1]["theta"] = -1.0
        curves[1]["forward"] = -1.0
        coverage[0]["k"] = float("nan")
        public_nodes = surface.nodes; public_nodes[1]["theta"] = -1.0
        public_curves = surface.curves; public_curves[1]["forward"] = -1.0
        self.assertEqual(surface.total_variance(0.0, data["nodes"][1]["t"]), expected)
        with self.assertRaises(AttributeError):
            surface.rho = 1.01

    def test_artifact_hash_units_and_rehashed_constraints_are_revalidated(self):
        artifact = oracle_surface().to_dict()
        self.assertEqual(artifact["curve_sha256"], canonical_hash(artifact["curves"]))
        self.assertEqual(artifact["surface_sha256"], canonical_hash(
            {key: item for key, item in artifact.items() if key != "surface_sha256"}))
        for field, value in (("volatility_unit", "percent"), ("premium_unit", "USD"),
                             ("time_convention", "ACT/360"), ("parameterisation", "unrestricted_svi")):
            edited = deepcopy(artifact); edited[field] = value; refresh_outer_hash(edited)
            with self.subTest(field=field), self.assertRaises(ValueError):
                VolSurface.from_dict(edited)
        edited = deepcopy(artifact); edited["curves"][0]["forward"] *= 1.001
        with self.assertRaises(ValueError):
            VolSurface.from_dict(edited)  # Outer hash catches positive carry edits.
        refresh_outer_hash(edited)
        with self.assertRaises(ValueError):
            VolSurface.from_dict(edited)  # Curve hash independently catches them.
        for field, value in (("rho", 1.01), ("eta", 3.0)):
            edited = deepcopy(artifact); edited[field] = value; refresh_outer_hash(edited)
            with self.subTest(field=field), self.assertRaises(ValueError):
                VolSurface.from_dict(edited)  # A recomputed digest cannot bypass constraints.
        edited = deepcopy(artifact); edited["curves"][0]["discount_factor"] = -1.0
        edited["curve_sha256"] = canonical_hash(edited["curves"]); refresh_outer_hash(edited)
        with self.assertRaises(ValueError):
            VolSurface.from_dict(edited)
        for metadata in ({"curve_sha256": "0"*64}, {"as_of": "2026-10-05"},
                         {"source": {"as_of": "2026-10-05"}}):
            edited = deepcopy(artifact); edited["metadata"] = metadata; refresh_outer_hash(edited)
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                VolSurface.from_dict(edited)


class SurfaceFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validated = validate_snapshot(quotes())
        cls.fit = fit_ssvi(cls.validated, starts=3, maxiter=1000, seed=42, diagnostics=False)

    def test_recovers_independent_synthetic_prices_and_constraints(self):
        self.assertEqual(self.fit["status"], "converged")
        self.assertIsNotNone(self.fit["surface"])
        surface = VolSurface.from_dict(self.fit["surface"])
        self.assertEqual(len(self.fit["qualifying_expiries"]), 4)
        self.assertLess(self.fit["train"]["price_rmse"], 1e-4)
        self.assertLess(self.fit["holdout"]["price_rmse"], 1e-4)
        for case in reference()["cases"]:
            if abs(case["k"]) <= 0.14:
                self.assertAlmostEqual(surface.implied_vol(case["strike"], case["expiry"]),
                                       case["expected_iv"], delta=1e-4)

    def test_held_out_quotes_and_vendor_surface_never_choose_parameters(self):
        altered = deepcopy(self.validated)
        for row in altered["accepted"]:
            if row["split"] == "holdout":
                row["close"] *= 1.2
                row["iv"] *= 1.3
                row["uncertainty_price"] *= 9.0
                row["vendor_iv"] = 7.0
        altered["vendor_surface"] = {"points": [{"iv": 7.0, "strike": 100.0}]}
        repeated = fit_ssvi(altered, starts=3, maxiter=1000, seed=42, diagnostics=False)
        self.assertEqual(repeated["parameters"], self.fit["parameters"])
        self.assertEqual(repeated["chosen_start"], self.fit["chosen_start"])
        self.assertEqual(repeated["qualifying_expiries"], self.fit["qualifying_expiries"])
        self.assertNotEqual(repeated["holdout"]["price_rmse"], self.fit["holdout"]["price_rmse"])

    def test_minimum_expiry_and_training_group_rules(self):
        reduced = deepcopy(self.validated)
        keep = set(sorted({row["expiry"] for row in reduced["accepted"]})[:2])
        reduced["accepted"] = [row for row in reduced["accepted"] if row["expiry"] in keep]
        result = fit_ssvi(reduced, starts=1, maxiter=20, diagnostics=False)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertIsNone(result["surface"])
        reduced = deepcopy(self.validated)
        expiry = sorted({row["expiry"] for row in reduced["accepted"]})[-1]
        train_groups = sorted({row["group"] for row in reduced["accepted"] if row["expiry"] == expiry and row["split"] == "train"})[:4]
        reduced["accepted"] = [row for row in reduced["accepted"] if row["expiry"] != expiry or row["split"] == "holdout" or row["group"] in train_groups]
        result = fit_ssvi(reduced, starts=1, maxiter=500, diagnostics=False)
        self.assertEqual(len(result["qualifying_expiries"]), 3)
        self.assertIn(expiry, [item["expiry"] for item in result["excluded_expiries"]])
        self.assertNotIn(expiry, [node["expiry"] for node in result["surface"]["nodes"]])

    def test_valid_surface_with_conflicting_quotes_preserves_model_residual(self):
        altered = deepcopy(self.validated)
        target = next(row for row in altered["accepted"] if row["split"] == "train"
                      and row["kind"] == ("call" if row["strike"] >= row["forward"] else "put"))
        target["close"] += 2.0
        fit = fit_ssvi(altered, starts=1, maxiter=1000, diagnostics=False)
        self.assertIn(fit["status"], ("converged", "incomplete"))
        self.assertIsNotNone(fit["surface"])
        VolSurface.from_dict(fit["surface"])
        self.assertGreater(fit["train"]["price_rmse"], 1e-3)
        predictions = [row for row in fit["predictions"] if row["model_price"] is not None]
        self.assertTrue(any(abs(row["price_error"]) > 1e-3 for row in predictions))

    def test_whole_expiry_holdout_removes_the_held_node(self):
        fit = fit_ssvi(self.validated, starts=1, maxiter=500, diagnostics=True)
        self.assertTrue(fit["expiry_holdouts"])
        self.assertEqual(len(fit["sensitivities"]), 6)
        self.assertEqual(sum(entry["category"] == "carry" for entry in fit["sensitivities"]), 4)
        self.assertEqual(sum(entry["category"] == "weight" for entry in fit["sensitivities"]), 2)
        for fold in fit["expiry_holdouts"]:
            expiry = fold["held_out_expiry"]
            self.assertNotIn(expiry, fold["training_expiries"])
            if fold["surface"] is not None:
                self.assertNotIn(expiry, [node["expiry"] for node in fold["surface"]["nodes"]])
        interior = [fold for fold in fit["expiry_holdouts"]
                    if fold["held_out_expiry"] in (reference()["nodes"][1]["expiry"], reference()["nodes"][2]["expiry"])]
        self.assertEqual(len(interior), 2)
        self.assertTrue(all(fold["holdout"]["count"] > 0 for fold in interior))

    def test_one_iteration_never_reports_a_converged_solver(self):
        fit = fit_ssvi(self.validated, starts=1, maxiter=1, diagnostics=False)
        self.assertNotEqual(fit["status"], "converged")
        self.assertIn(fit["status"], ("incomplete", "failed"))

    def test_solver_objective_gradient_matches_independent_finite_differences(self):
        import numpy as np
        from scipy.optimize import minimize
        checked = []

        def verify_and_minimize(objective, initial, *args, **kwargs):
            vector = np.array(initial, dtype=float, copy=True)
            loss, gradient = objective(vector)
            self.assertTrue(math.isfinite(loss))
            self.assertEqual(len(gradient), len(vector))
            for index in range(len(vector)):
                spacing = 1e-5 * (1 + abs(vector[index]))
                up, down = vector.copy(), vector.copy()
                up[index] += spacing; down[index] -= spacing
                estimate = (objective(up)[0] - objective(down)[0]) / (2*spacing)
                self.assertTrue(math.isclose(float(gradient[index]), estimate, rel_tol=2e-5, abs_tol=1e-7),
                                (index, float(gradient[index]), estimate))
            checked.append(True)
            return minimize(objective, initial, *args, **kwargs)

        with patch("scipy.optimize.minimize", side_effect=verify_and_minimize):
            fit_ssvi(self.validated, starts=1, maxiter=50, diagnostics=False)
        self.assertEqual(checked, [True])

    def test_dividend_scenarios_keep_original_price_targets_and_isolate_failure(self):
        import tools.market_validation.surface as surface_module
        original_fit_core = surface_module._fit_core
        original = deepcopy(self.validated)
        by_ric = {row["ric"]: row for row in original["accepted"]}
        representatives = {row["ric"] for row in original["accepted"]
                           if row["kind"] == ("call" if row["strike"] >= row["forward"] else "put")}
        q_scenarios, base = [], []

        def verify_scenario_and_fit(scenario, **kwargs):
            first = scenario["accepted"][0]
            change = first["q"] - by_ric[first["ric"]]["q"]
            if change:
                self.assertTrue(math.isclose(abs(change), 0.0001, rel_tol=1e-10))
                self.assertEqual(kwargs["representative_rics"], representatives)
                self.assertEqual({row["ric"] for row in scenario["accepted"]}, set(by_ric))
                for row in scenario["accepted"]:
                    before = by_ric[row["ric"]]
                    for field in ("close", "strike", "expiry", "ric", "group", "split", "uncertainty_price", "r", "discount_factor"):
                        self.assertEqual(row[field], before[field], field)
                    self.assertAlmostEqual(row["q"], before["q"] + change, places=14)
                    self.assertAlmostEqual(row["forward"], before["forward"] * math.exp(-change*row["t"]), places=12)
                q_scenarios.append(change)
                if change > 0:
                    raise ValueError("independent test: dividend scenario evaluation failed")
            result = original_fit_core(scenario, **kwargs)
            if "representative_rics" not in kwargs:
                base.append(result["parameters"])
            return result

        with patch("tools.market_validation.surface._fit_core", side_effect=verify_scenario_and_fit):
            fit = fit_ssvi(self.validated, starts=1, maxiter=500, diagnostics=True)
        self.assertEqual(self.validated, original)
        self.assertEqual(fit["parameters"], base[0])
        self.assertEqual(len(q_scenarios), 2)
        self.assertEqual(len(fit["sensitivities"]), 6)
        failed = [entry for entry in fit["sensitivities"] if entry["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("dividend scenario evaluation failed", failed[0]["error"])

    def test_dividend_scenario_invalid_price_bounds_are_reported(self):
        # This valid ITM call has enough time value for r+1bp, but not for
        # q-1bp. It is an unselected call/put counterpart, so changing it
        # must neither select new targets nor alter the baseline fit.
        raw = quotes()
        target = raw["quotes"][0]
        curve = raw["curves"][0]
        t = (date.fromisoformat(target["expiry"]) - date.fromisoformat(raw["source"]["as_of"])).days/365.0
        discount, forward, strike = curve["discount_factor"], curve["forward"], target["strike"]
        after_q_minus = discount * (forward * math.exp(0.0001*t) - strike)
        after_r_plus = discount * math.exp(-0.0001*t) * (forward * math.exp(0.0001*t) - strike)
        target["close"] = (after_q_minus + after_r_plus)/2
        validated = validate_snapshot(raw)
        self.assertTrue(any(row["ric"] == target["ric"] for row in validated["accepted"]))
        before = deepcopy(validated)
        baseline = fit_ssvi(validated, starts=1, maxiter=500, diagnostics=False)
        fit = fit_ssvi(validated, starts=1, maxiter=500, diagnostics=True)
        self.assertEqual(fit["parameters"], baseline["parameters"])
        self.assertEqual(validated, before)
        self.assertEqual(fit["status"], "converged")
        q_minus = next(entry for entry in fit["sensitivities"] if entry["name"] == "dividend_minus_1bp")
        self.assertEqual(q_minus["status"], "failed")
        self.assertIn(target["ric"], [row["ric"] for row in q_minus["invalid_quotes"]])
        r_plus = next(entry for entry in fit["sensitivities"] if entry["name"] == "risk_free_plus_1bp")
        self.assertEqual(r_plus["status"], "converged")


class RealLsegSurfaceReplayTests(unittest.TestCase):
    def test_actual_lseg_snapshots_have_fixed_dates_hashes_and_group_splits(self):
        for filename, as_of, accepted_count in (
            ("lseg_spxw_20261006_surface.json", "2026-10-06", 72),
            ("lseg_spxw_20261008_surface.json", "2026-10-08", 84),
        ):
            with self.subTest(snapshot=filename):
                data = json.loads((FIXTURES / filename).read_text())
                self.assertEqual(data["source"]["provider"], "LSEG")
                self.assertEqual(data["source"]["as_of"], as_of)
                self.assertEqual(data["dataset_sha256"], dataset_hash(data))
                validated = validate_snapshot(data)
                self.assertEqual(validated["dataset_sha256"], data["dataset_sha256"])
                self.assertEqual(len(validated["accepted"]), accepted_count)
                splits = {}
                for row in validated["accepted"]:
                    self.assertEqual(row["as_of"], as_of)
                    self.assertEqual(row["close_date"], as_of)
                    self.assertGreater(row["discount_factor"], 0)
                    self.assertGreater(row["forward"], 0)
                    self.assertEqual(splits.setdefault(row["group"], row["split"]), row["split"])

    def test_real_replay_fit_preserves_constraints_and_reports_market_error(self):
        for filename in ("lseg_spxw_20261006_surface.json", "lseg_spxw_20261008_surface.json"):
            with self.subTest(snapshot=filename):
                validated = validate_snapshot(json.loads((FIXTURES / filename).read_text()))
                fit = fit_ssvi(validated, starts=1, maxiter=1000, seed=42, diagnostics=False)
                self.assertIn(fit["status"], ("converged", "incomplete"))
                self.assertIsNotNone(fit["surface"])
                self.assertGreaterEqual(len(fit["qualifying_expiries"]), 3)
                for expiry in fit["qualifying_expiries"]:
                    training_groups = {row["group"] for row in validated["accepted"]
                                       if row["expiry"] == expiry and row["split"] == "train"}
                    self.assertGreaterEqual(len(training_groups), 5)
                surface = VolSurface.from_dict(fit["surface"])
                self.assertEqual(fit["dataset_sha256"], validated["dataset_sha256"])
                self.assertEqual(surface.metadata["dataset_sha256"], validated["dataset_sha256"])
                self.assertTrue(math.isfinite(fit["train"]["price_rmse"]))
                self.assertTrue(math.isfinite(fit["holdout"]["price_rmse"]))
                self.assertGreater(fit["train"]["price_rmse"], 1e-3)
                for prediction in fit["predictions"]:
                    if prediction["model_price"] is not None:
                        self.assertTrue(math.isfinite(prediction["model_price"]))
                        self.assertTrue(math.isfinite(prediction["price_error"]))
                for node in fit["surface"]["nodes"]:
                    for k in (-5.0, 0.0, 5.0):
                        d = surface.derivatives(k, node["t"], allow_wing=True)
                        density_factor = ((1 - k*d["w_k"]/(2*d["w"]))**2
                                          - 0.25*d["w_k"]**2*(1/d["w"] + 0.25) + 0.5*d["w_kk"])
                        self.assertGreaterEqual(density_factor, -1e-10)

    def test_invalid_curves_cannot_be_used_by_the_surface_fit(self):
        data = json.loads((FIXTURES / "lseg_spxw_20261008_surface.json").read_text())
        invalid_expiry = data["curves"][0]["expiry"]
        data["curves"][0]["as_of"] = "2026-10-07"
        validated = validate_snapshot(data)
        self.assertFalse(any(row["expiry"] == invalid_expiry for row in validated["accepted"]))
        self.assertTrue(validated["rejected"])


class SurfaceCliStatusTests(unittest.TestCase):
    def test_requested_surface_stage_succeeds_only_after_convergence(self):
        from tools.market_validation.__main__ import main
        for status in ("converged", "incomplete", "failed", "insufficient_data"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                candidate = {"status": status, "surface": oracle_surface().to_dict()
                             if status in ("converged", "incomplete") else None}
                with patch("tools.market_validation.surface.fit_ssvi", return_value=candidate), \
                     patch("tools.market_validation.report.write_report", return_value=Path(directory)/"report.html") as report, \
                     patch("tools.market_validation.lseg.fetch_snapshot", side_effect=AssertionError("offline replay contacted LSEG")), \
                     redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = main(["run", "--snapshot", str(FIXTURES/"synthetic_ssvi_quotes.json"),
                                   "--output", directory, "--surface-model", "ssvi", "--no-calibration", "--no-mc"])
                self.assertEqual(result, 0 if status == "converged" else 1)
                self.assertEqual(report.call_args.kwargs["surface_fit"]["status"], status)


if __name__ == "__main__":
    unittest.main()
