"""Standard-library smoke checks for native local-volatility context pricing."""

from datetime import date, timedelta
import json
import math
from pathlib import Path
import tempfile
import unittest

import fuzzy_enigma as fe


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "local_vol"
ISSUE = date(2026, 1, 1)


def day(n):
    return (ISSUE + timedelta(days=n)).isoformat()


def inputs(name="smile", *, forward_multiplier=1.0):
    artifact = json.loads((FIXTURES / (name + ".json")).read_text())
    rows = artifact["curves"]
    dates = [r["date"] for r in rows]
    discounts = [r["discount_factor"] for r in rows]
    forwards = [r["forward"] if i == 0 else r["forward"] * forward_multiplier
                for i, r in enumerate(rows)]
    discount = fe.DiscountCurve(artifact["as_of"], artifact["currency"], dates, discounts, "synthetic native fixture")
    forward = fe.ForwardCurve(artifact["as_of"], artifact["currency"], artifact["asset"], dates, forwards, "synthetic native fixture")
    context = fe.ValuationContext(fe.ValuationCutoff(artifact["as_of"]), artifact["spot"], artifact["currency"], artifact["asset"], discount, forward)
    return fe.LocalVolModel(json.dumps(artifact)), context, artifact


def snowball():
    return fe.DatedSnowballNote(100.0, 100.0, 0.12, 70.0, 110.0, ISSUE, "USD", "INDEX",
        [day(n) for n in (0,30,60,90)],
        fe.EventSchedule([day(60),day(90)],[day(65),day(95)],"explicit synthetic calendar"),
        fe.EventSchedule([day(90)],[day(95)],"explicit synthetic calendar"))


class LocalVolTests(unittest.TestCase):
    def test_flat_smile_limit_and_serialization(self):
        model, _, artifact = inputs("flat")
        for t in (0.0, 0.001, 90/365, 0.7, 1.0):
            for k in (-10.0, -0.3, 0.0, 0.2, 10.0):
                self.assertAlmostEqual(model.total_variance(k,t), 0.04*t, places=13)
                self.assertAlmostEqual(model.local_variance(k,t), 0.04, places=13)
        self.assertEqual(model.identity, artifact["artifact_sha256"])
        self.assertEqual(fe.LocalVolModel(model.to_json()).identity, model.identity)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"surface.json"
            target.write_text(model.to_json())
            self.assertEqual(fe.LocalVolModel.load(str(target)).identity, model.identity)
        with self.assertRaises(AttributeError):
            model.identity = "changed"

    def test_finite_origin_wings_and_time_bounds(self):
        model, _, _ = inputs()
        for k in (-1000.0,-2.0,-0.1,0.0,0.1,2.0,1000.0):
            for t in (0.0,1e-8,90/365,0.75,1.0):
                values = model.derivatives(k,t)
                self.assertTrue(all(math.isfinite(values[name]) for name in ("w","w_k","w_kk","w_t","g","local_variance")))
                self.assertGreater(values["g"], 1e-10)
                self.assertGreaterEqual(values["local_variance"],0.0)
        self.assertEqual(model.derivatives(0,0)["w"],0.0)
        for k,t in ((float("nan"),0.1),(0,-0.01),(0,1.0001)):
            with self.assertRaises(ValueError):
                model.local_variance(k,t)

    def test_artifact_tampering_and_unknown_fields_rejected(self):
        _, _, artifact = inputs()
        for key,value in (("rho",0.2),("fit_status","failed"),("schema_version",2),("wings","clamp")):
            bad=dict(artifact,**{key:value})
            with self.assertRaises(ValueError):
                fe.LocalVolModel(json.dumps(bad))
        bad=dict(artifact,unverified_field=True)
        with self.assertRaises(ValueError):
            fe.LocalVolModel(json.dumps(bad))

    def test_same_random_stream_flat_gbm_and_exact_diagnostic_counts(self):
        model, context, _ = inputs("flat")
        grid=fe.TimeGrid(ISSUE,[day(90)],max_step_days=3.0)
        engine=fe.McEngine(paths=129,seed=42,antithetic=True,parallel=True)
        lv=engine.price_european_context(context,model,"call",100.0,day(90),grid=grid)
        gbm=engine.price_european_context(context,fe.GbmDynamics(0.2),"call",100.0,day(90),grid=grid)
        self.assertAlmostEqual(lv.price,gbm.price,places=12)
        self.assertAlmostEqual(lv.std_error,gbm.std_error,places=12)
        self.assertEqual((lv.paths,lv.samples),(130,65))
        self.assertEqual(lv.path_diagnostics["evaluations"],130*grid.steps)
        self.assertEqual(lv.path_diagnostics["paths_with_extensions"],130)
        self.assertEqual(lv.path_diagnostics["short_time_evaluations"],130*grid.steps)
        self.assertIsNone(lv.state)
        self.assertIsNone(gbm.path_diagnostics)
        serial=fe.McEngine(paths=129,seed=42,antithetic=True,parallel=False).price_european_context(context,model,"call",100.0,day(90),grid=grid)
        self.assertEqual((lv.price,lv.std_error,lv.path_diagnostics),(serial.price,serial.std_error,serial.path_diagnostics))

    def test_native_put_call_and_barrier_in_out_parity(self):
        model, context, _ = inputs()
        engine=fe.McEngine(paths=256,seed=19)
        grid=fe.TimeGrid(ISSUE,[day(30),day(60),day(90)],max_step_days=2.0)
        call=engine.price_european_context(context,model,"call",100.0,day(90),grid=grid)
        put=engine.price_european_context(context,model,"put",100.0,day(90),grid=grid)
        parity=context.discount_curve.value(day(90))*(context.forward_curve.value(day(90))-100.0)
        self.assertLess(abs(call.price-put.price-parity),4*(call.std_error+put.std_error)+1e-10)
        out=engine.price_barrier_context(context,model,"call",100.0,day(90),"up_and_out",115.0,[day(0),day(30),day(60),day(90)],grid=grid)
        inside=engine.price_barrier_context(context,model,"call",100.0,day(90),"up_and_in",115.0,[day(0),day(30),day(60),day(90)],grid=grid)
        self.assertAlmostEqual(out.price+inside.price,call.price,places=12)
        self.assertAlmostEqual(sum(row["present_value"] for row in call.expected_cashflows),call.price,places=12)

    def test_notes_and_legacy_restrictions(self):
        model, context, _ = inputs()
        note=snowball()
        state=fe.replay_note_history(note,[],context.cutoff)
        engine=fe.McEngine(paths=256,seed=11)
        value=engine.price_snowball_dated(note,context,model,state)
        self.assertTrue(math.isfinite(value.price))
        self.assertEqual(value.state.contract_hash,state.contract_hash)
        self.assertIsNotNone(value.path_diagnostics)
        schedule=fe.EventSchedule([day(30),day(60),day(90)],[day(35),day(65),day(95)],"explicit synthetic calendar")
        phoenix=fe.DatedPhoenixNote(100.0,100.0,3.0,90.0,70.0,110.0,ISSUE,"USD","INDEX",
            [day(n) for n in (0,30,60,90)],schedule,schedule,
            fe.EventSchedule([day(90)],[day(95)],"explicit synthetic calendar"),memory=True)
        phoenix_state=fe.replay_note_history(phoenix,[],context.cutoff)
        phoenix_value=engine.price_phoenix_dated(phoenix,context,model,phoenix_state)
        self.assertTrue(math.isfinite(phoenix_value.price))
        self.assertIsNotNone(phoenix_value.path_diagnostics)
        for operation in (
            lambda: engine.price_european(model,"call",100.0),
            lambda: engine.greeks_european(model,"call",100.0),
            lambda: engine.price_asian(model,"call",100.0,control=True),
            lambda: engine.price_continuous_barrier(model,"call","up_and_out",100.0,120.0),
            lambda: engine.calendar_theta(note,context,model,state,day(1)),
        ):
            with self.assertRaises(ValueError):
                operation()

    def test_context_carry_and_date_coverage_failures(self):
        model, bad_context, _ = inputs(forward_multiplier=1.01)
        with self.assertRaises(ValueError):
            fe.McEngine(paths=64).price_european_context(bad_context,model,"call",100,day(90))
        _, context, _ = inputs()
        with self.assertRaises(ValueError):
            fe.McEngine(paths=64).price_european_context(context,model,"call",100,day(366))
        with self.assertRaises(ValueError):
            fe.McEngine(paths=64).price_european_context(context,model,"call",100,day(90),payment_date=day(89))


if __name__ == "__main__":
    unittest.main()
