"""Smoke tests for the Heston conditional Monte Carlo Python entry point."""

import math
import unittest

import fuzzy_enigma as fe


class ConditionalHestonTests(unittest.TestCase):
    def test_deterministic_variance_has_analytic_price_and_zero_sampling_error(self):
        model = fe.HestonModel(100., .03, .01, .04, 1.5, .04, 0., -.7, 1., steps=16)
        result = fe.McEngine(paths=7).price_heston_conditional(model, "call", 105.)
        expected = fe.black_scholes_price(fe.GbmModel(100., .03, .01, .2, 1., steps=16), "call", 105.)
        self.assertAlmostEqual(result.price, expected, places=11)
        self.assertEqual(result.std_error, 0.)
        self.assertEqual(result.samples, 4)

    def test_stochastic_variance_is_reproducible(self):
        model = fe.HestonModel(100., .03, .01, .04, 1.5, .04, .4, -.7, 1., steps=32)
        serial = fe.McEngine(paths=1001, seed=42, parallel=False).price_heston_conditional(model, "put", 95.)
        parallel = fe.McEngine(paths=1001, seed=42, parallel=True).price_heston_conditional(model, "put", 95.)
        self.assertEqual((serial.price, serial.std_error, serial.samples),
                         (parallel.price, parallel.std_error, parallel.samples))
        self.assertGreater(serial.price, 0.)
        self.assertTrue(math.isfinite(serial.std_error))
        self.assertEqual(serial.samples, 501)

    def test_model_type_payoff_and_path_budget_are_checked(self):
        engine = fe.McEngine(paths=2)
        model = fe.HestonModel(100., .03, .01, .04, 1.5, .04, .4, -.7, 1.)
        for wrong_model in (fe.GbmModel(100., .03, .01, .2, 1.), object(), None):
            with self.subTest(model=wrong_model), self.assertRaises(ValueError):
                engine.price_heston_conditional(wrong_model, "call", 100.)
        for kind, strike in (("bad", 100.), ("call", -1.), ("put", float("nan"))):
            with self.subTest(kind=kind, strike=strike), self.assertRaises(ValueError):
                engine.price_heston_conditional(model, kind, strike)
        for shifts in ([], [1.], [0., 0.], [0., float("nan")], [0., float("inf")]):
            with self.subTest(shifts=shifts), self.assertRaises(ValueError):
                engine.price_heston_conditional(model, "call", 100., variance_shifts=shifts)
        with self.assertRaises(ValueError):
            fe.McEngine(paths=0).price_heston_conditional(model, "call", 100.)


if __name__ == "__main__":
    unittest.main()
