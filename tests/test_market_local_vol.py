"""Offline smooth-export, native-evaluator and independent PDE regressions.

Tolerances are fixed below before the benchmark is evaluated. Saved market
data are fitted independently on each run; no supplier session is opened.
"""

from datetime import date, timedelta
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fuzzy_enigma as fe
import QuantLib as ql

from tools.market_validation import (
    SmoothLocalVolSurface, VolSurface, context_from_snapshot, export_local_vol,
    finite_difference_price, fit_ssvi,
)
from tools.market_validation.analytics import validate_snapshot
from tools.market_validation.local_vol import REFERENCE_TOLERANCES, _pchip, artifact_hash
from tools.market_validation.snapshot import read_snapshot


FIXTURES = ROOT / "tests/fixtures/local_vol"
MARKET = ROOT / "tests/fixtures/market"
# Independent PDE acceptance budgets, chosen before computing a benchmark.
FLAT_PDE_ABS = REFERENCE_TOLERANCES["flat_pde_price_abs_spot"]*100
MARKET_PDE_SPOT_FRACTION = REFERENCE_TOLERANCES["market_pde_price_abs_spot"]
MARKET_REFINEMENT_SPOT_FRACTION = REFERENCE_TOLERANCES["market_pde_refinement_abs_spot"]
MC_SIGMA_MULTIPLE = REFERENCE_TOLERANCES["mc_sigma_multiple"]
MC_GRID_SPOT_FRACTION = REFERENCE_TOLERANCES["mc_discretisation_abs_spot"]


def fixture(name="smile"):
    return SmoothLocalVolSurface.load(FIXTURES / f"{name}.json")


def source_and_context(name="smile"):
    model = fixture(name)
    value = model.to_dict()
    rows = value["curves"]
    dates = [row["date"] for row in rows]
    discount = fe.DiscountCurve(model.as_of, "USD", dates,
                                [row["discount_factor"] for row in rows], "synthetic reference discount")
    forward = fe.ForwardCurve(model.as_of, "USD", "INDEX", dates,
                             [row["forward"] for row in rows], "synthetic reference forward")
    context = fe.ValuationContext(fe.ValuationCutoff(model.as_of), 100, "USD", "INDEX", discount, forward)
    origin = date.fromisoformat(model.as_of)
    nodes = [{"expiry": (origin+timedelta(days=round(node["t"]*365))).isoformat(), "theta": node["theta"]}
             for node in value["nodes"][1:]]
    curves = [{"as_of": model.as_of, "expiry": row["date"], "discount_factor": row["discount_factor"],
               "forward": row["forward"], "source": "synthetic reference"} for row in rows[1:]]
    surface = VolSurface(model.as_of, model.rho, model.eta, nodes, curves, model.coverage)
    return surface, context


def rehash(value):
    value["artifact_sha256"] = artifact_hash(value)
    return value


class SmoothExportTests(unittest.TestCase):
    def test_variance_division_overflow_is_rejected_by_both_evaluators(self):
        artifact = fixture("flat").to_dict()
        times, values = [0.0, 4e-308], [0.0, 2.0]
        tangents, coefficients = _pchip(times, values)
        artifact.update(
            rho=0.0, eta=0.5, theta_floor=1e300,
            nodes=[{"t": t, "theta": value, "tangent": tangent}
                   for t, value, tangent in zip(times, values, tangents)],
            coefficients=coefficients,
            coverage=[{"k": k, "t": times[-1]} for k in (-0.2, 0.2)],
        )
        rehash(artifact)
        # A finite, accepted input can overflow only when dividing w_t by g.
        # Both implementations must reject that query instead of returning inf.
        smooth = SmoothLocalVolSurface(artifact)
        native = fe.LocalVolModel(json.dumps(artifact))
        for model in (smooth, native):
            with self.assertRaisesRegex(ValueError, "non-finite|nonfinite"):
                model.local_variance(6e150, 0.0)

    def test_export_preserves_old_surface_and_records_model_changes(self):
        original, context = source_and_context()
        before = original.to_dict()
        smooth = original.export_local_vol(context=context, allow_synthetic=True)
        self.assertEqual(original.to_dict(), before)
        self.assertEqual(smooth.to_dict(), fixture().to_dict())
        self.assertEqual(smooth.to_dict()["source_fit_status"], "synthetic_reference")
        self.assertTrue(smooth.export_report["quote_surface_unchanged"])
        self.assertGreater(smooth.export_report["max_abs_iv_change_bp"], 0)
        self.assertGreater(smooth.export_report["max_abs_call_price_change"], 0)
        self.assertTrue(any(row["t"] not in [node["t"] for node in original.nodes]
                            for row in smooth.export_report["differences"]))

    def test_fit_status_gate_cannot_promote_failed_market_data(self):
        original, context = source_and_context()
        with self.assertRaisesRegex(ValueError, "converged"):
            export_local_vol(original, context=context)
        for status in ("incomplete", "failed", "insufficient_data"):
            invalid = VolSurface(original.as_of, original.rho, original.eta,
                                 original.nodes, original.curves, original.coverage,
                                 metadata={"fit_status": status})
            with self.assertRaisesRegex(ValueError, "converged"):
                export_local_vol(invalid, context=context, allow_synthetic=True)
        with self.assertRaisesRegex(ValueError, "Boolean"):
            export_local_vol(original, context=context, allow_synthetic=1)

    def test_curve_and_asset_consistency_are_mandatory(self):
        original, context = source_and_context()
        missing = fe.ValuationContext(context.cutoff, 100, "USD", "INDEX", context.discount_curve)
        with self.assertRaisesRegex(ValueError, "forward"):
            export_local_vol(original, context=missing, allow_synthetic=True)
        dates = context.forward_curve.dates
        wrong = fe.ForwardCurve(original.as_of, "USD", "INDEX", dates,
                                [100]+[value*1.01 for value in context.forward_curve.values[1:]], "wrong carry")
        altered = fe.ValuationContext(context.cutoff, 100, "USD", "INDEX", context.discount_curve, wrong)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            export_local_vol(original, context=altered, allow_synthetic=True)
        dated = VolSurface(original.as_of, original.rho, original.eta, original.nodes, original.curves,
                           original.coverage, metadata={"source": {"as_of": original.as_of, "underlying_ric": "OTHER"}})
        with self.assertRaisesRegex(ValueError, "assets"):
            export_local_vol(dated, context=context, allow_synthetic=True)

    def test_payment_lag_does_not_require_forward_after_stock_support(self):
        original, context = source_and_context()
        discount = fe.DiscountCurve(original.as_of, "USD", context.discount_curve.dates+["2027-02-01"],
                                   context.discount_curve.values+[0.96], "payment lag discount")
        lagged = fe.ValuationContext(context.cutoff, 100, "USD", "INDEX", discount, context.forward_curve)
        smooth = export_local_vol(original, context=lagged, allow_synthetic=True)
        self.assertEqual(smooth.to_dict()["curves"][-1]["date"], "2027-01-01")

    def test_pchip_coefficients_match_independent_scipy(self):
        from scipy.interpolate import PchipInterpolator
        scenarios = [[0, .02, .025, .1], [0, .02, .02, .1], [0, 0, .02, .03], [0, .4, .401, .402]]
        times = [0, .2, .5, 1]
        for values in scenarios:
            slopes, coefficients = _pchip(times, values)
            independent = PchipInterpolator(times, values)
            for t, slope in zip(times, slopes):
                self.assertAlmostEqual(slope, float(independent.derivative()(t)), places=14)
            for i, row in enumerate(coefficients):
                for u in (.1, .5, .9):
                    t = times[i]+u*(times[i+1]-times[i])
                    theta = row["a"]+u*(row["b"]+u*(row["c"]+u*row["d"]))
                    self.assertAlmostEqual(theta, float(independent(t)), places=14)

    def test_independent_quantlib_raw_svi_total_variance(self):
        smooth = fixture()
        for t in (.0001, .1, .2465753424657534, .6, 1):
            theta, _ = smooth._theta(t)
            phi = smooth.eta/math.sqrt((theta+smooth.theta_floor)*(1+theta))
            _, forward = smooth._carry(t)
            raw = [.5*theta*(1-smooth.rho*smooth.rho), .5*theta*phi,
                   math.sqrt(1-smooth.rho*smooth.rho)/phi, smooth.rho, -smooth.rho/phi]
            reference = ql.SviSmileSection(t, forward, raw)
            for k in (-1, -.2, 0, .2, 1):
                expected = float(reference.variance(forward*math.exp(k)))
                self.assertTrue(math.isclose(smooth.total_variance(k, t), expected,
                                             abs_tol=1e-12, rel_tol=1e-10))

    def test_analytic_derivatives_against_finite_differences(self):
        smooth = fixture()
        for k in (-.2, -.05, .03, .2):
            for t in (.12, .37, .74):
                row = smooth.derivatives(k, t)
                hk, ht = 1e-5, 1e-6
                center = smooth.total_variance(k, t)
                left, right = smooth.total_variance(k-hk, t), smooth.total_variance(k+hk, t)
                first = (right-left)/(2*hk)
                second = (right-2*center+left)/(hk*hk)
                time_first = (smooth.total_variance(k, t+ht)-smooth.total_variance(k, t-ht))/(2*ht)
                self.assertTrue(math.isclose(row["w_k"], first, abs_tol=1e-10, rel_tol=1e-6))
                self.assertTrue(math.isclose(row["w_kk"], second, abs_tol=1e-7, rel_tol=1e-4))
                self.assertTrue(math.isclose(row["w_t"], time_first, abs_tol=1e-9, rel_tol=1e-6))

    def test_time_knots_are_c1_and_origin_limit_is_finite(self):
        smooth = fixture()
        for node in smooth.nodes[1:-1]:
            t = node["t"]
            for k in (-.2, 0, .2):
                exact = smooth.derivatives(k, t)
                for side in (-1, 1):
                    adjacent = smooth.derivatives(k, t+side*1e-9)
                    self.assertTrue(math.isclose(exact["w_t"], adjacent["w_t"], abs_tol=1e-8, rel_tol=1e-6))
        for k in (-5, -.2, 0, .2, 5):
            origin = smooth.derivatives(k, 0)
            near = smooth.derivatives(k, 1e-11)
            self.assertEqual(origin["w"], 0)
            self.assertGreaterEqual(origin["g"], .25)
            self.assertTrue(math.isclose(origin["local_variance"], near["local_variance"], abs_tol=1e-8, rel_tol=1e-6))

    def test_constant_volatility_and_independent_black_limit(self):
        smooth = fixture("flat")
        normal = lambda x: math.erfc(-x/math.sqrt(2))/2
        for t in (0, .01, .25, .5, 1):
            for k in (-2, 0, 2):
                self.assertAlmostEqual(smooth.local_variance(k, t), .04, places=13)
                self.assertAlmostEqual(smooth.total_variance(k, t), .04*t, places=13)
            if t:
                discount, forward = smooth._carry(t)
                strike = 105
                sd = .2*math.sqrt(t)
                d1 = math.log(forward/strike)/sd+sd/2
                expected = discount*(forward*normal(d1)-strike*normal(d1-sd))
                self.assertTrue(math.isclose(smooth.black_price("call", strike, t), expected,
                                             abs_tol=1e-9, rel_tol=1e-10))

    def test_zero_variance_and_origin_plateau_have_no_division_fallback(self):
        value = fixture("flat").to_dict()
        for node in value["nodes"]:
            node["theta"] = node["tangent"] = 0
        for row in value["coefficients"]:
            row.update(a=0, b=0, c=0, d=0)
        smooth = SmoothLocalVolSurface(rehash(value))
        for t in (0, .1, 1):
            self.assertEqual(smooth.local_variance(.4, t), 0)
            self.assertEqual(smooth.total_variance(.4, t), 0)
            self.assertEqual(smooth.derivatives(.4, t)["g"], 1)

    def test_global_constraint_domain_and_query_guards(self):
        smooth = fixture()
        for t in (0, .00001, .2, .5, 1):
            for k in (-100, -10, -1, 0, 1, 10, 100):
                row = smooth.derivatives(k, t)
                self.assertGreater(row["g"], 1e-10)
                self.assertGreaterEqual(row["local_variance"], 0)
                self.assertTrue(math.isfinite(row["local_variance"]))
        for t in (-.1, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                smooth.local_variance(0, t)
        self.assertTrue(smooth.derivatives(10, .5)["outside_observed_coverage"])
        self.assertFalse(smooth.derivatives(0, .5)["outside_observed_coverage"])
        self.assertLess(smooth.total_variance(.2, .4), smooth.total_variance(.2, .6))
        with self.assertRaises(ValueError):
            smooth.implied_vol(100, 0)

    def test_artifact_and_report_serialization_are_separate_and_hashed(self):
        original, context = source_and_context()
        smooth = export_local_vol(original, context=context, allow_synthetic=True)
        with tempfile.TemporaryDirectory() as folder:
            path = smooth.save(Path(folder)/"surface.json")
            self.assertEqual(SmoothLocalVolSurface.load(path).to_dict(), smooth.to_dict())
            report = json.loads(path.with_suffix(".export_report.json").read_text())
            self.assertEqual(report["artifact_sha256"], smooth.to_dict()["artifact_sha256"])
            self.assertEqual(report["report_type"], "local_vol_smooth_export")
        copied = smooth.nodes
        copied[0]["theta"] = 99
        self.assertEqual(smooth.nodes[0]["theta"], 0)

    def test_tampered_artifacts_are_rejected_even_with_a_recomputed_hash(self):
        cases = [lambda v: v.update(rho=.9995), lambda v: v.update(eta=2),
                 lambda v: v.update(theta_floor=0), lambda v: v.update(epsilon=0), lambda v: v.update(epsilon=1),
                 lambda v: v.update(min_denominator=0), lambda v: v.update(min_denominator=.25),
                 lambda v: v.update(min_denominator=.2499), lambda v: v.update(fit_status="incomplete"),
                 lambda v: v.update(source_fit_status="failed"), lambda v: v.update(coverage=[]),
                 lambda v: v["nodes"][1].update(theta=5),
                 lambda v: v["nodes"][1].update(tangent=.99),
                 lambda v: v["coefficients"][0].update(c=.99),
                 lambda v: v["curves"][0].update(forward=101),
                 lambda v: v["coverage"][0].update(t=.001)]
        for change in cases:
            value = fixture().to_dict()
            change(value)
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SmoothLocalVolSurface(rehash(value))
                with self.assertRaises(ValueError):
                    fe.LocalVolModel(json.dumps(value))
        for change in (lambda v: v.update(extra="ignored"), lambda v: v.update(schema_version=True),
                       lambda v: v.update(schema_version=1.0),
                       lambda v: v.update(artifact_sha256="0"*64)):
            value = fixture().to_dict()
            change(value)
            with self.assertRaises(ValueError):
                SmoothLocalVolSurface(value)
            with self.assertRaises(ValueError):
                fe.LocalVolModel(json.dumps(value))

    def test_python_native_evaluation_and_validation_agree(self):
        for name in ("flat", "smile"):
            smooth = fixture(name)
            native = fe.LocalVolModel(json.dumps(smooth.to_dict()))
            self.assertEqual(native.identity, smooth.to_dict()["artifact_sha256"])
            for t in (0, 1e-11, .1, smooth.nodes[1]["t"], .4, .9, 1):
                for k in (-20, -.25, 0, .25, 20):
                    python = smooth.derivatives(k, t)
                    rust = native.derivatives(k, t)
                    for field in ("theta", "theta_t", "w", "w_k", "w_kk", "w_t", "g", "local_variance"):
                        self.assertTrue(math.isclose(python[field], rust[field], abs_tol=1e-12, rel_tol=1e-10),
                                        msg=(name, k, t, field, python[field], rust[field]))
            for key, bad in (("rho", .9995), ("theta_floor", 0), ("epsilon", 1), ("fit_status", "incomplete")):
                invalid = smooth.to_dict()
                invalid[key] = bad
                rehash(invalid)
                with self.assertRaises(ValueError):
                    fe.LocalVolModel(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    SmoothLocalVolSurface(invalid)


class IndependentPdeTests(unittest.TestCase):
    def test_mc_flat_and_smile_prices_against_independent_vanilla_reference(self):
        engine = fe.McEngine(paths=65536, seed=42)
        for name in ("flat", "smile"):
            smooth = fixture(name)
            _original, context = source_and_context(name)
            native = fe.LocalVolModel(json.dumps(smooth.to_dict()))
            expiry = "2026-06-30"
            for strike in (95, 110):
                with self.subTest(model=name, strike=strike):
                    values = []
                    reference = smooth.black_price("call", strike, expiry)
                    for max_step in (1, .125):
                        grid = fe.TimeGrid(smooth.as_of, [expiry], max_step_days=max_step)
                        result = engine.price_european_context(context, native, "call", strike, expiry, grid=grid)
                        self.assertLess(abs(result.price-reference), MC_SIGMA_MULTIPLE*result.std_error+MC_GRID_SPOT_FRACTION*100)
                        values.append(result)
                    combined_se = math.hypot(values[0].std_error, values[1].std_error)
                    self.assertLess(abs(values[0].price-values[1].price), MC_SIGMA_MULTIPLE*combined_se+MC_GRID_SPOT_FRACTION*100)

    def test_flat_model_recovers_independent_black_prices(self):
        smooth = fixture("flat")
        for strike in (90, 100, 110):
            for kind in ("call", "put"):
                result = finite_difference_price(smooth, kind, strike, .5,
                                                 space_points=401, time_steps=400)
                self.assertLess(abs(result["price_error"]), FLAT_PDE_ABS)
                self.assertEqual(result["generator"], "0.5 * local_variance * (dxx - dx)")

    def test_refined_pde_put_call_parity_and_space_boundary_diagnostics(self):
        smooth = fixture()
        call = finite_difference_price(smooth, "call", 100, .5, space_points=401, time_steps=400)
        put = finite_difference_price(smooth, "put", 100, .5, space_points=401, time_steps=400)
        discount, forward = smooth._carry(.5)
        self.assertAlmostEqual(call["price"]-put["price"], discount*(forward-100), delta=.002)
        self.assertEqual(call["x_bounds"], [-2, 2])
        self.assertIn("truncated", call["boundary_policy"])
        for kwargs in ({"space_points": True}, {"time_steps": 2}, {"x_bounds": (0, 2)}):
            with self.assertRaises(ValueError):
                finite_difference_price(smooth, "call", 100, .5, **kwargs)


class MarketLocalVolReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.markets = []
        for day in ("20261006", "20261008"):
            snapshot = read_snapshot(MARKET / f"lseg_spxw_{day}_surface.json")
            validated = validate_snapshot(snapshot)
            fit = fit_ssvi(validated, starts=8, seed=42, maxiter=2000, diagnostics=False)
            if fit["status"] != "converged":
                raise AssertionError("market fit is not accepted: "+fit["status"])
            context, audit = context_from_snapshot(snapshot, currency="USD")
            original = VolSurface.from_dict(fit["surface"])
            smooth = export_local_vol(original, context=context)
            cls.markets.append((snapshot, original, smooth, fit, audit))

    def test_both_frozen_markets_bind_dates_carry_hashes_and_original_fit(self):
        for snapshot, original, smooth, fit, audit in self.markets:
            with self.subTest(as_of=smooth.as_of):
                value = smooth.to_dict()
                self.assertEqual(value["source_dataset_sha256"], snapshot["dataset_sha256"])
                self.assertEqual(value["source_surface_sha256"], fit["surface_sha256"])
                self.assertEqual(value["source_curve_sha256"], fit["curve_sha256"])
                self.assertEqual(value["as_of"], audit["as_of"])
                self.assertEqual(value["currency"], "USD")
                self.assertEqual(value["asset"], ".SPX")
                self.assertEqual(value["curves"][0]["forward"], snapshot["spot"])
                self.assertEqual(original.to_dict(), fit["surface"])
                self.assertEqual(value["theta_floor"], .01*original.nodes[0]["theta"])
                self.assertTrue(all(math.isfinite(row["price_change"]) for row in smooth.export_report["differences"]))
                self.assertGreater(smooth.diagnostics()["minimum_density_denominator"], 1e-10)
                native = fe.LocalVolModel(json.dumps(value))
                self.assertTrue(math.isclose(native.local_variance(-.1, .1), smooth.local_variance(-.1, .1),
                                             abs_tol=1e-12, rel_tol=1e-10))

    def test_independent_pde_reprices_market_surface_and_reports_grid_change(self):
        for snapshot, _original, smooth, _fit, _audit in self.markets:
            spot = snapshot["spot"]
            for multiplier in (.9, 1, 1.1):
                with self.subTest(as_of=smooth.as_of, moneyness=multiplier):
                    coarse = finite_difference_price(smooth, "call", spot*multiplier, .25,
                                                     space_points=401, time_steps=400)
                    refined = finite_difference_price(smooth, "call", spot*multiplier, .25,
                                                      space_points=801, time_steps=800)
                    self.assertLess(abs(refined["price_error"]), MARKET_PDE_SPOT_FRACTION*spot)
                    self.assertLess(abs(refined["price_error"]), abs(coarse["price_error"]))
                    self.assertLess(abs(refined["price"]-coarse["price"]), MARKET_REFINEMENT_SPOT_FRACTION*spot)
                    self.assertEqual(refined["artifact_sha256"], smooth.to_dict()["artifact_sha256"])

    def test_refined_mc_reprices_both_market_surfaces_and_records_extensions(self):
        engine = fe.McEngine(paths=65536, seed=42)
        for snapshot, _original, smooth, _fit, _audit in self.markets:
            context, _audit = context_from_snapshot(snapshot, currency="USD")
            native = fe.LocalVolModel(json.dumps(smooth.to_dict()))
            expiry = (date.fromisoformat(smooth.as_of)+timedelta(days=90)).isoformat()
            spot = snapshot["spot"]
            reference = smooth.black_price("call", spot, expiry)
            values = []
            for max_step in (1, .125):
                with self.subTest(as_of=smooth.as_of, max_step_days=max_step):
                    grid = fe.TimeGrid(smooth.as_of, [expiry], max_step_days=max_step)
                    result = engine.price_european_context(context, native, "call", spot, expiry, grid=grid)
                    self.assertLess(abs(result.price-reference), MC_SIGMA_MULTIPLE*result.std_error+MC_GRID_SPOT_FRACTION*spot)
                    diagnostic = result.path_diagnostics
                    self.assertEqual(diagnostic["evaluations"], grid.steps*65536)
                    self.assertGreater(diagnostic["outside_observed_evaluations"], 0)
                    self.assertGreater(diagnostic["short_time_evaluations"], 0)
                    self.assertIn("global_shifted_phi", diagnostic["extension_policy"])
                    values.append(result)
            combined_se = math.hypot(values[0].std_error, values[1].std_error)
            self.assertLess(abs(values[0].price-values[1].price), MC_SIGMA_MULTIPLE*combined_se+MC_GRID_SPOT_FRACTION*spot)


if __name__ == "__main__":
    unittest.main()
