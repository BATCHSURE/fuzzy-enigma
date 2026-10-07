"""Smoke and contract checks for the structured/numerical/Heston bindings.

Run with ``python tests/test_extensions.py`` after building the extension.
Only the standard library is required; the Rust suite checks convergence
against independent pricing references in more detail.
"""

import math
import sys

import fuzzy_enigma as fe


SNOWBALL = dict(
    notional=100.0,
    reference_spot=100.0,
    coupon_rate=0.12,
    knock_in_barrier=70.0,
    knock_out_barrier=105.0,
    observation_indices=[12, 24, 36, 48],
)
PHOENIX = dict(
    notional=100.0,
    reference_spot=100.0,
    coupon_per_period=3.0,
    coupon_barrier=90.0,
    knock_in_barrier=70.0,
    knock_out_barrier=105.0,
    coupon_indices=[12, 24, 36, 48],
    autocall_indices=[24, 48],
)
HESTON = dict(
    spot=100.0,
    r=0.03,
    q=0.0,
    v0=0.0625,
    kappa=2.0,
    theta=0.04,
    xi=0.35,
    rho=-0.65,
    t=1.0,
    steps=48,
)


def close(actual, expected, tolerance=1e-9):
    assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)


def value_error(call, description):
    try:
        call()
    except ValueError:
        return
    except BaseException as exc:
        raise AssertionError(
            f"{description}: expected ValueError, received {type(exc).__name__}"
        ) from exc
    raise AssertionError(f"{description}: no exception raised")


def finite_price(result, expected_samples):
    assert math.isfinite(result.price) and result.price >= 0.0, result
    assert math.isfinite(result.std_error) and result.std_error >= 0.0, result
    assert result.samples == expected_samples, result
    low, high = result.confidence_95()
    assert low <= result.price <= high, result


def finite_greeks(greeks):
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert math.isfinite(getattr(greeks, name)), (name, greeks)


def test_heston_prices_every_supported_payoff():
    model = fe.HestonModel(**HESTON)
    for name, expected in HESTON.items():
        assert getattr(model, name) == expected, name
    assert "HestonModel" in repr(model)
    engine = fe.McEngine(paths=10_000, seed=11)
    results = {
        "european": engine.price_european(model, "call", 100.0),
        "asian": engine.price_asian(model, "call", 100.0),
        "asian_scheduled": engine.price_asian_scheduled(
            model, "call", 100.0, [12, 24, 36, 48]
        ),
        "barrier": engine.price_barrier(model, "call", "up_and_out", 100.0, 130.0),
        "lookback": engine.price_lookback(model, "call"),
        "cliquet": engine.price_cliquet(model, 100.0, -0.03, 0.05),
        "autocallable": engine.price_autocallable(
            model, 100.0, 3.0, 105.0, 70.0, [12, 24, 36, 48]
        ),
        "snowball": engine.price_snowball(model, **SNOWBALL),
        "phoenix": engine.price_phoenix(model, **PHOENIX, memory=True),
    }
    for result in results.values():
        finite_price(result, 5_000)
    serial = fe.McEngine(paths=10_000, seed=11, parallel=False)
    serial_results = {
        "european": serial.price_european(model, "call", 100.0),
        "phoenix": serial.price_phoenix(model, **PHOENIX, memory=True),
    }
    for name, result in serial_results.items():
        parallel = results[name]
        assert (result.price, result.std_error, result.samples) == (
            parallel.price,
            parallel.std_error,
            parallel.samples,
        ), name


def test_heston_parameter_boundaries():
    invalid = {
        "spot": [0.0, -1.0, math.nan, math.inf],
        "r": [math.nan, math.inf],
        "q": [math.nan, -math.inf],
        "v0": [-0.01, math.nan, math.inf],
        "kappa": [0.0, -1.0, math.nan, math.inf],
        "theta": [-0.01, math.nan, math.inf],
        "xi": [-0.01, math.nan, math.inf],
        "rho": [-1.01, 1.01, math.nan, math.inf],
        "t": [0.0, -1.0, math.nan, math.inf],
        "steps": [0],
    }
    for name, values in invalid.items():
        for value in values:
            terms = {**HESTON, name: value}
            value_error(lambda: fe.HestonModel(**terms), f"Heston {name}={value}")
    engine = fe.McEngine(paths=32, seed=11)
    for rho in (-1.0, 1.0):
        model = fe.HestonModel(
            100.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, rho, 1.0, steps=4
        )
        result = engine.price_european(model, "call", 95.0)
        close(result.price, 5.0)
        close(result.std_error, 0.0)


def test_gbm_only_methods_reject_heston():
    model = fe.HestonModel(**HESTON)
    engine = fe.McEngine(paths=32, seed=11)
    calls = {
        "legacy Asian European control": lambda: engine.price_asian(
            model, "call", 100.0, control=True
        ),
        "legacy barrier European control": lambda: engine.price_barrier(
            model, "call", "up_and_out", 100.0, 130.0, control=True
        ),
        "scheduled Asian European control": lambda: engine.price_asian_scheduled(
            model, "call", 100.0, [12, 24], control="european"
        ),
        "scheduled Asian geometric control": lambda: engine.price_asian_scheduled(
            model, "call", 100.0, [12, 24], control="geometric"
        ),
        "geometric Asian analytic price": lambda: fe.geometric_asian_price(
            model, "call", 100.0, [12, 24]
        ),
        "continuous barrier": lambda: engine.price_continuous_barrier(
            model, "call", "up_and_out", 100.0, 130.0
        ),
        "European Greeks": lambda: engine.greeks_european(model, "call", 100.0),
        "Snowball Greeks": lambda: engine.greeks_snowball(model, **SNOWBALL),
        "Phoenix Greeks": lambda: engine.greeks_phoenix(model, **PHOENIX),
        "scheduled Asian Greeks": lambda: engine.greeks_asian_scheduled(
            model, "call", 100.0, [12, 24]
        ),
        "continuous barrier Greeks": lambda: engine.greeks_continuous_barrier(
            model, "call", "up_and_out", 100.0, 130.0
        ),
    }
    for description, call in calls.items():
        value_error(call, description)
    value_error(
        lambda: engine.price_european(object(), "call", 100.0),
        "unsupported model type",
    )


def test_structure_fixed_reference_and_new_greeks():
    engine = fe.McEngine(paths=128, seed=11)
    snowball = {
        **SNOWBALL,
        "knock_in_barrier": 95.0,
        "observation_indices": [2, 4],
    }
    phoenix = {
        **PHOENIX,
        "knock_in_barrier": 95.0,
        "coupon_per_period": 0.0,
        "coupon_indices": [1, 2, 3, 4],
        "autocall_indices": [2, 4],
    }
    for spot in (80.0, 90.0):
        model = fe.GbmModel(spot, 0.0, 0.0, 0.0, 1.0, steps=4)
        close(engine.price_snowball(model, **snowball).price, spot)
        close(engine.price_phoenix(model, **phoenix).price, spot)
        for greeks in (
            engine.greeks_snowball(model, **snowball),
            engine.greeks_phoenix(model, **phoenix),
        ):
            finite_greeks(greeks)
            close(greeks.delta, 1.0)
            close(greeks.gamma, 0.0)
            close(greeks.theta, 0.0)
    flat = fe.GbmModel(100.0, 0.0, 0.0, 0.0, 1.0, steps=48)
    close(engine.price_snowball(flat, **SNOWBALL).price, 112.0)
    close(engine.price_phoenix(flat, **PHOENIX).price, 112.0)
    greeks = engine.greeks_snowball(flat, **SNOWBALL)
    finite_greeks(greeks)
    close(greeks.delta, 0.0)
    close(greeks.theta, 12.0)
    # An eligible maturity coupon is independent of the maturity loss in par.
    impaired = fe.GbmModel(90.0, 0.0, 0.0, 0.0, 1.0, steps=48)
    close(
        engine.price_phoenix(
            impaired, **{**PHOENIX, "knock_in_barrier": 95.0}
        ).price,
        102.0,
    )


def test_new_payoff_validation():
    model = fe.GbmModel(100.0, 0.03, 0.0, 0.25, 1.0, steps=48)
    engine = fe.McEngine(paths=32, seed=11)
    for schedule in ([], [0, 12], [24, 12], [12, 12], [49]):
        value_error(
            lambda: engine.price_snowball(
                model, **{**SNOWBALL, "observation_indices": schedule}
            ),
            f"Snowball schedule {schedule}",
        )
        for name in ("coupon_indices", "autocall_indices"):
            value_error(
                lambda: engine.price_phoenix(model, **{**PHOENIX, name: schedule}),
                f"Phoenix {name} {schedule}",
            )
        for control in ("none", "european", "geometric"):
            value_error(
                lambda: engine.price_asian_scheduled(
                    model, "call", 100.0, schedule, control=control
                ),
                f"Asian {schedule} with {control}",
            )
        value_error(
            lambda: fe.geometric_asian_price(model, "call", 100.0, schedule),
            f"analytic Asian schedule {schedule}",
        )
    for terms, price in (
        (SNOWBALL, engine.price_snowball),
        (PHOENIX, engine.price_phoenix),
    ):
        positive_names = (
            "notional",
            "reference_spot",
            "knock_in_barrier",
            "knock_out_barrier",
        )
        if "coupon_barrier" in terms:
            positive_names += ("coupon_barrier",)
        for name in positive_names:
            for invalid in (0.0, -1.0, math.nan, math.inf):
                value_error(
                    lambda: price(model, **{**terms, name: invalid}),
                    f"{price.__name__} {name}={invalid}",
                )
        coupon = "coupon_rate" if "coupon_rate" in terms else "coupon_per_period"
        for invalid in (-0.01, math.nan, math.inf):
            value_error(
                lambda: price(model, **{**terms, coupon: invalid}),
                f"{price.__name__} {coupon}={invalid}",
            )
        finite_price(price(model, **{**terms, coupon: 0.0}), 16)
    for control in ("unknown", "True"):
        value_error(
            lambda: engine.price_asian_scheduled(
                model, "call", 100.0, [12, 24], control=control
            ),
            f"unsupported control {control}",
        )
    for strike in (0.0, -1.0, math.nan, math.inf):
        value_error(
            lambda: engine.price_asian_scheduled(model, "call", strike, [12, 24]),
            f"Asian strike {strike}",
        )
        value_error(
            lambda: engine.price_continuous_barrier(
                model, "call", "up_and_out", strike, 130.0
            ),
            f"continuous strike {strike}",
        )
    for barrier in (0.0, -1.0, math.nan, math.inf):
        value_error(
            lambda: engine.price_continuous_barrier(
                model, "call", "up_and_out", 100.0, barrier
            ),
            f"continuous barrier {barrier}",
        )


def test_asian_controls_and_closed_form_greeks():
    model = fe.GbmModel(100.0, 0.03, 0.0, 0.25, 1.0, steps=48)
    engine = fe.McEngine(paths=10_000, seed=19)
    dates = [12, 24, 36, 48]
    plain = engine.price_asian_scheduled(model, "call", 100.0, dates)
    european = engine.price_asian_scheduled(
        model, "call", 100.0, dates, control="european"
    )
    geometric = engine.price_asian_scheduled(
        model, "call", 100.0, dates, control="geometric"
    )
    assert geometric.std_error < 0.25 * plain.std_error, (plain, geometric)
    assert european.std_error < plain.std_error, (plain, european)
    for controlled in (european, geometric):
        assert abs(controlled.price - plain.price) < 4.0 * plain.std_error
    assert fe.geometric_asian_price(model, "call", 100.0, dates) > 0.0

    # One terminal fixing makes the arithmetic payoff identical to both
    # controls. Its price and bumped Greeks therefore have almost no MC
    # noise even with 64 paths. This catches stale control expectations.
    tiny_engine = fe.McEngine(paths=64, seed=19)
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    d1 = (0.03 + 0.25**2 / 2.0) / 0.25
    d2 = d1 - 0.25
    density = math.exp(-d1**2 / 2.0) / math.sqrt(2.0 * math.pi)
    expected_price = 100.0 * cdf(d1) - 100.0 * math.exp(-0.03) * cdf(d2)
    analytic_greeks = {
        "delta": (cdf(d1), 0.001),
        "gamma": (density / (100.0 * 0.25), 0.0001),
        "vega": (100.0 * density, 0.03),
        "theta": (
            100.0 * density * 0.25 / 2.0
            + 0.03 * 100.0 * math.exp(-0.03) * cdf(d2),
            0.03,
        ),
        "rho": (100.0 * math.exp(-0.03) * cdf(d2), 0.01),
    }
    close(fe.geometric_asian_price(model, "call", 100.0, [48]), expected_price, 1e-4)
    for control in ("european", "geometric"):
        result = tiny_engine.price_asian_scheduled(
            model, "call", 100.0, [48], control=control
        )
        close(result.price, expected_price, 1e-4)
        assert result.std_error < 1e-10, result
        greeks = tiny_engine.greeks_asian_scheduled(
            model, "call", 100.0, [48], control=control
        )
        finite_greeks(greeks)
        for name, (expected, tolerance) in analytic_greeks.items():
            close(getattr(greeks, name), expected, tolerance)


def test_continuous_barrier_parity_and_special_cases():
    model = fe.GbmModel(100.0, 0.03, 0.0, 0.25, 1.0, steps=48)
    engine = fe.McEngine(paths=10_000, seed=23)
    european = engine.price_european(model, "call", 100.0)
    for direction, barrier in (("up", 130.0), ("down", 80.0)):
        knock_out = engine.price_continuous_barrier(
            model, "call", f"{direction}_and_out", 100.0, barrier
        )
        knock_in = engine.price_continuous_barrier(
            model, "call", f"{direction}_and_in", 100.0, barrier
        )
        close(knock_out.price + knock_in.price, european.price)
        discrete_out = engine.price_barrier(
            model, "call", f"{direction}_and_out", 100.0, barrier
        )
        assert knock_out.price <= discrete_out.price + 1e-10
    initial_hit = engine.price_continuous_barrier(
        model, "call", "up_and_out", 100.0, 100.0, rebate=3.0
    )
    close(initial_hit.price, 3.0 * math.exp(-0.03))
    initial_in = engine.price_continuous_barrier(
        model, "call", "up_and_in", 100.0, 100.0
    )
    close(initial_in.price, european.price)
    flat = fe.GbmModel(100.0, 0.0, 0.0, 0.0, 1.0, steps=4)
    close(
        engine.price_continuous_barrier(
            flat, "call", "up_and_out", 95.0, 105.0
        ).price,
        5.0,
    )
    close(
        engine.price_continuous_barrier(
            flat, "call", "up_and_in", 95.0, 105.0, rebate=3.0
        ).price,
        3.0,
    )
    finite_greeks(
        fe.McEngine(paths=2_000, seed=23).greeks_continuous_barrier(
            model, "call", "up_and_out", 100.0, 130.0
        )
    )


def main():
    tests = (
        test_heston_prices_every_supported_payoff,
        test_heston_parameter_boundaries,
        test_gbm_only_methods_reject_heston,
        test_structure_fixed_reference_and_new_greeks,
        test_new_payoff_validation,
        test_asian_controls_and_closed_form_greeks,
        test_continuous_barrier_parity_and_special_cases,
    )
    for test in tests:
        test()
        print(f"  ok   {test.__name__}")
    print(f"all {len(tests)} extension test groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
