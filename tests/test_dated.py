"""Standard-library smoke and lifecycle checks for the native dated APIs."""

from datetime import date, datetime, timedelta
import math
import unittest

import fuzzy_enigma as fe


ISSUE = date(2026, 1, 1)


def day(n):
    return ISSUE + timedelta(days=n)


def schedule(fix, pay=None):
    return fe.EventSchedule([day(n) for n in fix], [day(n) for n in (pay if pay is not None else fix)], "explicit teaching calendar")


def snowball():
    return fe.DatedSnowballNote(
        100.0, 100.0, 0.12, 70.0, 105.0, ISSUE, "USD", "INDEX",
        [day(n) for n in (0, 10, 30, 60, 90)], schedule([30, 60, 90], [35, 65, 95]), schedule([90], [95]),
    )


def phoenix():
    return fe.DatedPhoenixNote(
        100.0, 100.0, 3.0, 90.0, 70.0, 105.0, ISSUE, "USD", "INDEX",
        [day(n) for n in (0, 30, 60, 90)], schedule([30, 60, 90], [35, 65, 95]),
        schedule([60, 90], [65, 95]), schedule([90], [95]), memory=True,
    )


def context(n=0, spot=100.0, r=0.03, carry=0.01, phase="after_fixing", forward=True):
    start, end = day(n), day(120)
    t = (end - start).days / 365.0
    discount = fe.DiscountCurve(start, "USD", [start, end], [1.0, math.exp(-r*t)], "synthetic flat discount")
    fwd = fe.ForwardCurve(start, "USD", "INDEX", [start, end], [spot, spot*math.exp(carry*t)], "synthetic flat carry") if forward else None
    return fe.ValuationContext(fe.ValuationCutoff(start, phase), spot, "USD", "INDEX", discount, fwd, "dated smoke fixture")


class DatedTests(unittest.TestCase):
    def test_date_only_curves_and_bounds(self):
        ctx = context(r=-0.02)
        self.assertGreater(ctx.discount_curve.value(day(30)), 1.0)
        self.assertAlmostEqual(ctx.discount_curve.value_at_time(30/365), math.exp(0.02*30/365), places=13)
        self.assertEqual(ctx.discount_curve.as_of, ISSUE.isoformat())
        self.assertEqual(ctx.forward_curve.dates, [ISSUE.isoformat(), day(120).isoformat()])
        for value in (datetime(2026, 1, 1), "2026-1-1", "2026-02-29", "2026-01-01T00:00:00"):
            with self.assertRaises(ValueError):
                fe.ValuationCutoff(value)
        with self.assertRaises(ValueError):
            ctx.discount_curve.value(day(121))
        with self.assertRaises(ValueError):
            fe.DiscountCurve(ISSUE, "USD", [ISSUE, day(10)], [0.99, 0.98], "bad origin")
        with self.assertRaises(ValueError):
            fe.ValuationContext(fe.ValuationCutoff(ISSUE), 101.0, "USD", "INDEX", ctx.discount_curve, ctx.forward_curve)
        with self.assertRaises(AttributeError):
            ctx.spot = 101.0

    def test_grid_preserves_fixings_and_date_interfaces(self):
        grid = fe.TimeGrid(ISSUE.isoformat(), [day(30), day(30), day(90)], max_step_days=2.0)
        fine = fe.TimeGrid(ISSUE, [day(30), day(90)], max_step_days=0.5)
        self.assertEqual(grid.fixing_dates, fine.fixing_dates)
        self.assertGreater(fine.steps, grid.steps)
        self.assertTrue(all(b > a for a, b in zip(grid.times, grid.times[1:])))
        self.assertAlmostEqual(grid.times[-1], 90/365)
        for dt in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                fe.TimeGrid(ISSUE, [day(90)], dt)
        with self.assertRaises(ValueError):
            schedule([30], [29])

    def test_redeemed_pending_partial_and_settled(self):
        note = snowball()
        state = fe.replay_note_history(note, [(day(10), 100.0), (day(30), 105.0)], fe.ValuationCutoff(day(30)))
        self.assertEqual(state.termination, "redeemed")
        engine = fe.McEngine(paths=0)
        value = engine.price_snowball_dated(note, context(30, forward=False), None, state)
        amount = 100.0 * (1 + 0.12*30/365)
        self.assertAlmostEqual(value.price, amount*math.exp(-0.03*5/365), places=10)
        self.assertEqual((value.std_error, value.samples, value.method), (0.0, 0, "deterministic"))
        with self.assertRaises(ValueError):
            fe.advance_note_state(note, state, [], fe.ValuationCutoff(day(36)))
        partial = []
        for item in state.ledger:
            paid = 40.0 if item["kind"] == "principal" else item["amount"]
            partial.append(fe.SettlementConfirmation(item["id"], day(36),
                [fe.ActualPayment("receipt-"+item["kind"], day(35), paid)],
                day(40) if paid < item["amount"] else None))
        partial_state = fe.advance_note_state(note, state, [], fe.ValuationCutoff(day(36)), settlements=partial)
        value = engine.price_snowball_dated(note, context(36, forward=False), None, partial_state)
        self.assertAlmostEqual(value.price, 60.0*math.exp(-0.03*4/365), places=10)
        principal = next(item for item in partial_state.ledger if item["kind"] == "principal")
        paid = fe.SettlementConfirmation(principal["id"], day(40),
            [fe.ActualPayment("receipt-principal", day(35), 40.0), fe.ActualPayment("balance", day(40), 60.0)])
        settled = fe.advance_note_state(note, partial_state, [], fe.ValuationCutoff(day(40)), settlements=[paid])
        value = engine.price_snowball_dated(note, context(40, forward=False), None, settled)
        self.assertEqual((value.price, value.std_error, value.samples), (0.0, 0.0, 0))

    def test_memory_receivables_and_mc_reproducibility(self):
        note = phoenix()
        state = fe.replay_note_history(note, [(day(30), 80.0)], fe.ValuationCutoff(day(30)))
        self.assertEqual(len(state.memory_coupon_ids), 1)
        state = fe.advance_note_state(note, state, [(day(60), 95.0)], fe.ValuationCutoff(day(60)))
        self.assertEqual(len(state.memory_coupon_ids), 0)
        self.assertAlmostEqual(sum(item["outstanding_amount"] for item in state.known_receivables), 6.0)
        ctx = context(60, spot=95.0, r=0.0, carry=0.0)
        engine = fe.McEngine(paths=128, seed=7)
        result = engine.price_phoenix_dated(note, ctx, fe.GbmDynamics(0.0), state)
        self.assertAlmostEqual(result.price, 109.0)
        self.assertAlmostEqual(sum(row["present_value"] for row in result.expected_cashflows), result.price)
        heston = fe.HestonDynamics(0.04, 2.0, 0.04, 0.3, -0.6)
        parallel = engine.price_phoenix_dated(note, ctx, heston, state)
        serial = fe.McEngine(paths=128, seed=7, parallel=False).price_phoenix_dated(note, ctx, heston, state)
        self.assertEqual((parallel.price, parallel.std_error, parallel.samples), (serial.price, serial.std_error, serial.samples))
        with self.assertRaises(ValueError):
            engine.price_phoenix_dated(note, ctx, fe.GbmModel(95.0, 0.0, 0.0, 0.2, 1.0, 90), state)
        with self.assertRaises(ValueError):
            engine.price_phoenix_dated(note, context(61, 95.0), heston, state)

    def test_before_fixing_requires_explicit_scenario(self):
        note = snowball()
        cutoff = fe.ValuationCutoff(day(30), "before_fixing")
        state = fe.replay_note_history(note, [(day(10), 100.0)], cutoff)
        with self.assertRaises(ValueError):
            fe.McEngine(paths=64).price_snowball_dated(note, context(30, phase="before_fixing"), fe.GbmDynamics(0.2), state)
        scenario = fe.replay_note_history(note, [(day(10), 100.0)], cutoff,
            same_day_fixing_scenario=(day(30), 105.0))
        self.assertEqual(scenario.provenance, "fixing_scenario")
        result = fe.McEngine(paths=0).price_snowball_dated(note, context(30, phase="before_fixing", forward=False), None, scenario)
        self.assertEqual(result.samples, 0)

    def test_calendar_theta_payment_confirmation_and_pairing(self):
        note = snowball()
        state = fe.replay_note_history(note, [(day(10), 100.0), (day(30), 105.0)], fe.ValuationCutoff(day(30)))
        ctx = context(30, forward=False)
        engine = fe.McEngine(paths=0)
        roll = engine.calendar_theta(note, ctx, None, state, day(31))
        self.assertGreater(roll.pv_change, 0.0)
        self.assertEqual(roll.std_error, 0.0)
        self.assertAlmostEqual(roll.cash_adjusted_change, 0.0, places=10)
        with self.assertRaises(ValueError):
            engine.calendar_theta(note, ctx, None, state, day(35))
        settlements = [fe.SettlementConfirmation(item["id"], day(35),
            [fe.ActualPayment("paid-"+item["kind"], day(35), item["amount"])]) for item in state.ledger]
        paid = engine.calendar_theta(note, ctx, None, state, day(35), settlements=settlements)
        self.assertAlmostEqual(paid.cash_paid, sum(item["amount"] for item in state.ledger))
        self.assertAlmostEqual(paid.cash_adjusted_change, 0.0, places=10)
        fresh = fe.replay_note_history(note, [], fe.ValuationCutoff(ISSUE))
        stochastic = fe.McEngine(paths=256, seed=9).calendar_theta(note, context(), fe.GbmDynamics(0.2), fresh, day(1))
        self.assertTrue(math.isfinite(stochastic.pv_change))
        self.assertGreaterEqual(stochastic.std_error, 0.0)
        self.assertEqual(stochastic.roll_days, 1)
        serial = fe.McEngine(paths=256, seed=9, parallel=False).calendar_theta(note, context(), fe.GbmDynamics(0.2), fresh, day(1))
        self.assertEqual((stochastic.pv_change, stochastic.std_error), (serial.pv_change, serial.std_error))

    def test_declared_scenarios_survive_frozen_roll_and_remain_auditable(self):
        note = snowball()
        cutoff = fe.ValuationCutoff(day(30))
        scenario = fe.replay_note_history(note, [(day(10), 100.0)], cutoff,
            same_day_fixing_scenario=(day(30), 105.0))
        alternative = fe.replay_note_history(note, [(day(10), 100.0)], cutoff,
            same_day_fixing_scenario=(day(30), 100.0))
        self.assertNotEqual(scenario.history_hash, alternative.history_hash)
        self.assertEqual(scenario.same_day_fixing_scenario, (day(30).isoformat(), 105.0))
        engine = fe.McEngine(paths=0)
        first = engine.calendar_theta(note, context(30, forward=False), None, scenario, day(31))
        self.assertEqual(first.rolled_state.provenance, "frozen_roll")
        self.assertEqual(first.rolled_state.scenario_fixings, [(day(30).isoformat(), 105.0)])
        second = engine.calendar_theta(note, context(31, forward=False), None, first.rolled_state, day(32))
        self.assertAlmostEqual(second.cash_adjusted_change, 0.0, places=10)
        self.assertEqual(second.rolled_state.scenario_fixings, first.rolled_state.scenario_fixings)
        with self.assertRaises(ValueError):
            fe.advance_note_state(note, scenario, [], fe.ValuationCutoff(day(31)))


if __name__ == "__main__":
    unittest.main()
