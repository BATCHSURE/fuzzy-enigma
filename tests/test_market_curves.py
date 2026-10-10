"""Offline snapshot-to-native curve regressions; no supplier session required."""

from copy import deepcopy
from datetime import date, timedelta
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import fuzzy_enigma as fe

from tools.market_validation import context_from_snapshot
from tools.market_validation.context import context_from_snapshot as bridge
from tools.market_validation.snapshot import dataset_hash, read_snapshot

FIXTURES = ROOT / "tests/fixtures/market"
CAPTURES = [FIXTURES / "lseg_spxw_20261006_surface.json",
            FIXTURES / "lseg_spxw_20261008_surface.json"]


def data():
    return read_snapshot(CAPTURES[-1])


def rehash(snapshot):
    snapshot["dataset_sha256"] = dataset_hash(snapshot)
    return snapshot


class SnapshotCurveTests(unittest.TestCase):
    def test_public_export(self):
        self.assertIs(context_from_snapshot, bridge)

    def test_both_real_captures_preserve_dated_spot_and_every_node(self):
        for path in CAPTURES:
            with self.subTest(capture=path.name):
                snapshot = read_snapshot(path)
                context, audit = bridge(path, currency="USD")
                as_of = snapshot["source"]["as_of"]
                self.assertEqual(str(context.cutoff.as_of), as_of)
                self.assertEqual(context.cutoff.phase, "after_fixing")
                self.assertEqual(context.spot, snapshot["spot"])
                self.assertEqual(context.currency, "USD")
                self.assertEqual(context.asset, ".SPX")
                self.assertEqual(context.discount_curve.value(as_of), 1.0)
                self.assertEqual(context.forward_curve.value(as_of), snapshot["spot"])
                for row in snapshot["curves"]:
                    self.assertEqual(context.discount_curve.value(row["expiry"]), row["discount_factor"])
                    self.assertEqual(context.forward_curve.value(row["expiry"]), row["forward"])
                self.assertEqual(audit["parent_dataset_sha256"], snapshot["dataset_sha256"])
                self.assertEqual(audit["parent_capture_dataset_sha256"],
                                 snapshot["source"]["capture_dataset_sha256"])
                self.assertNotEqual(audit["derived_curve_sha256"], audit["parent_dataset_sha256"])
                self.assertEqual(audit["captured_nodes"][0]["discount_factor"],
                                 snapshot["curves"][0]["discount_factor"])
                self.assertEqual(audit["source"], snapshot["source"])

    def test_independent_log_interpolation_including_derived_origin(self):
        snapshot = data()
        context, _ = bridge(snapshot, currency="USD")
        start = date.fromisoformat(snapshot["source"]["as_of"])
        nodes = [(start, 1.0, snapshot["spot"])] + [
            (date.fromisoformat(row["expiry"]), row["discount_factor"], row["forward"])
            for row in snapshot["curves"]]
        for left, right in zip(nodes, nodes[1:]):
            days = (right[0] - left[0]).days
            query = left[0] + timedelta(days=days // 2)
            weight = (query - left[0]).days / days
            expected_d = math.exp((1.0 - weight)*math.log(left[1]) + weight*math.log(right[1]))
            expected_f = math.exp((1.0 - weight)*math.log(left[2]) + weight*math.log(right[2]))
            self.assertTrue(math.isclose(context.discount_curve.value(query), expected_d,
                                         rel_tol=2e-15, abs_tol=2e-15))
            self.assertTrue(math.isclose(context.forward_curve.value(query), expected_f,
                                         rel_tol=3e-15, abs_tol=1e-10))

    def test_hashes_are_stable_and_inputs_are_not_mutated(self):
        snapshot = data()
        original = deepcopy(snapshot)
        _, first = bridge(snapshot, currency="USD")
        _, second = bridge(snapshot, currency="USD")
        self.assertEqual(snapshot, original)
        self.assertEqual(first, second)
        reordered = deepcopy(snapshot)
        reordered["curves"].reverse()
        _, third = bridge(rehash(reordered), currency="USD")
        self.assertEqual(first["derived_curve_sha256"], third["derived_curve_sha256"])
        self.assertNotEqual(first["parent_dataset_sha256"], third["parent_dataset_sha256"])

    def test_grid_hash_records_actual_numerical_grid_and_checks_coverage(self):
        snapshot = data()
        as_of = snapshot["source"]["as_of"]
        fixings = [row["expiry"] for row in snapshot["curves"][:3]]
        coarse = fe.TimeGrid(as_of, fixings, max_step_days=7)
        refined = fe.TimeGrid(as_of, fixings, max_step_days=1)
        coarse_context, a = bridge(snapshot, currency="USD", time_grid=coarse)
        refined_context, b = bridge(snapshot, currency="USD", time_grid=refined)
        self.assertEqual(a["derived_curve_sha256"], b["derived_curve_sha256"])
        self.assertNotEqual(a["time_grid_sha256"], b["time_grid_sha256"])
        self.assertNotEqual(a["context_sha256"], b["context_sha256"])
        self.assertEqual(a["time_grid"]["times"], list(coarse.times))
        self.assertEqual(coarse_context.grid.identity, coarse.identity)
        self.assertEqual(refined_context.grid.identity, refined.identity)
        with self.assertRaisesRegex(ValueError, "as_of"):
            bridge(snapshot, currency="USD", time_grid=fe.TimeGrid("2026-10-09", fixings))
        with self.assertRaisesRegex(ValueError, "coverage"):
            bridge(snapshot, currency="USD", time_grid=fe.TimeGrid(as_of, ["2027-10-01"]))

    def test_supplied_snapshot_hash_is_required_and_verified(self):
        snapshot = data()
        for bad in [None, "", "a"*64, "not-a-hash"]:
            with self.subTest(hash=bad):
                broken = deepcopy(snapshot)
                broken["dataset_sha256"] = bad
                with self.assertRaisesRegex(ValueError, "dataset_sha256"):
                    bridge(broken, currency="USD")
        snapshot["spot"] += 1.0
        with self.assertRaisesRegex(ValueError, "does not match"):
            bridge(snapshot, currency="USD")

    def test_currency_is_explicit_and_conflicting_metadata_fails(self):
        for currency in [None, "", "usd", "US", "USDD"]:
            with self.subTest(currency=currency):
                with self.assertRaisesRegex(ValueError, "currency"):
                    bridge(data(), currency=currency)
        with self.assertRaises(TypeError):
            bridge(data())
        for location in ["quote", "curve", "source"]:
            snapshot = data()
            if location == "quote":
                snapshot["quotes"][0]["currency"] = "EUR"
            elif location == "curve":
                snapshot["curves"][0]["currency"] = "EUR"
            else:
                snapshot["source"]["currency"] = "EUR"
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "currency"):
                bridge(rehash(snapshot), currency="USD")

    def test_invalid_dated_nodes_and_sources_fail(self):
        changes = [
            lambda s: s["curves"].append(deepcopy(s["curves"][0])),
            lambda s: s["curves"][0].update(as_of="2026-10-07"),
            lambda s: s["curves"][0].update(expiry="2026-10-08"),
            lambda s: s["curves"][0].update(expiry="2026-02-30"),
            lambda s: s["curves"][0].update(discount_factor=0.0),
            lambda s: s["curves"][0].update(forward=-1.0),
            lambda s: s["curves"][0].update(source=""),
            lambda s: s["source"].update(underlying_ric=""),
            lambda s: s.update(curves=[]),
        ]
        for index, change in enumerate(changes):
            snapshot = data()
            change(snapshot)
            with self.subTest(case=index), self.assertRaises(ValueError):
                bridge(rehash(snapshot), currency="USD")

    def test_nonfinite_input_fails_even_before_native_construction(self):
        for field in ["spot", "discount_factor", "forward"]:
            for value in [float("nan"), float("inf")]:
                snapshot = data()
                if field == "spot":
                    snapshot[field] = value
                else:
                    snapshot["curves"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "nonfinite"):
                    bridge(snapshot, currency="USD")

    def test_eod_close_cannot_be_used_as_a_before_fixing_context(self):
        with self.assertRaisesRegex(ValueError, "end-of-day"):
            bridge(data(), currency="USD", phase="before_fixing")
        with self.assertRaisesRegex(ValueError, "phase"):
            bridge(data(), currency="USD", phase="midday")

    def test_curve_domain_is_bounded_and_negative_rates_are_valid(self):
        snapshot = data()
        for index, row in enumerate(snapshot["curves"]):
            row["discount_factor"] = 1.001 + 0.001*index
        context, _ = bridge(rehash(snapshot), currency="USD")
        self.assertGreater(context.discount_curve.value(snapshot["curves"][0]["expiry"]), 1.0)
        for curve in [context.discount_curve, context.forward_curve]:
            with self.assertRaises(ValueError):
                curve.value("2026-10-07")
            with self.assertRaises(ValueError):
                curve.value("2027-10-01")

    def test_bridge_needs_neither_option_inversion_nor_supplier_imports(self):
        snapshot = data()
        snapshot["quotes"] = []
        snapshot["vendor_surface"] = {"status": "not_used_by_curve_bridge"}
        with patch.dict(sys.modules, {"lseg": None, "lseg.data": None}), \
                patch("tools.market_validation.analytics.validate_snapshot", side_effect=AssertionError("quote inversion")):
            context, audit = bridge(rehash(snapshot), currency="USD")
        self.assertEqual(context.spot, snapshot["spot"])
        self.assertEqual(audit["source"]["provider"], "LSEG")

    def test_real_curve_discounts_a_synthetic_redeemed_note_and_frozen_roll(self):
        context, _ = bridge(data(), currency="USD")
        as_of = context.cutoff.as_of
        autocall = fe.EventSchedule([as_of], ["2026-10-15"], "synthetic unadjusted dates")
        maturity = fe.EventSchedule(["2027-03-31"], ["2027-04-07"], "synthetic unadjusted dates")
        note = fe.DatedSnowballNote(100.0, 8000.0, 0.12, 5600.0, context.spot,
            "2026-09-08", "USD", ".SPX", ["2026-09-08", as_of, "2027-03-31"],
            autocall, maturity)
        state = fe.replay_note_history(note, [(as_of, context.spot)], context.cutoff)
        engine = fe.McEngine(paths=0)
        value = engine.price_snowball_dated(note, context, None, state)
        elapsed = (date.fromisoformat(as_of) - date(2026, 9, 8)).days / 365.0
        amount = 100.0*(1.0 + 0.12*elapsed)
        independent_pv = amount*data()["curves"][0]["discount_factor"]
        self.assertAlmostEqual(value.price, independent_pv, places=11)
        self.assertEqual((value.method, value.samples, value.std_error), ("deterministic", 0, 0.0))
        roll = engine.calendar_theta(note, context, None, state, "2026-10-09")
        independent_rolled_pv = independent_pv / context.discount_curve.value("2026-10-09")
        self.assertAlmostEqual(roll.rolled_price, independent_rolled_pv, places=11)
        self.assertAlmostEqual(roll.pv_change, independent_rolled_pv - independent_pv, places=11)
        self.assertAlmostEqual(roll.cash_adjusted_change, 0.0, places=11)


if __name__ == "__main__":
    unittest.main()
