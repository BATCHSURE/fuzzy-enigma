"""Smoke tests for the PyO3 bindings.

Runs as a plain script (``python tests/test_bindings.py``) so CI needs no
test runner beyond the interpreter itself. The point is less to re-check
the numerics - the Rust suite does that - than to prove the extension
imports, prices, and reports bad input as a catchable ``ValueError``
instead of a panic escaping across the FFI boundary.
"""

import sys

import fuzzy_enigma as fe

FAILURES = []


def check(condition, description):
    if condition:
        print(f"  ok   {description}")
    else:
        print(f"  FAIL {description}")
        FAILURES.append(description)


def main():
    model = fe.GbmModel(spot=100, r=0.03, q=0.0, sigma=0.25, t=1.0, steps=252)
    engine = fe.McEngine(paths=50_000, seed=42, antithetic=True, parallel=True)

    print("pricing:")
    results = {
        "european": engine.price_european(model, "call", 100.0),
        "asian": engine.price_asian(model, "call", 100.0),
        "asian+cv": engine.price_asian(model, "call", 100.0, control=True),
        "barrier": engine.price_barrier(model, "call", "up_and_out", 100.0, 130.0),
        "lookback": engine.price_lookback(model, "call"),
        "cliquet": engine.price_cliquet(model, 100.0, -0.03, 0.05),
        "autocallable": engine.price_autocallable(
            model, 100.0, 2.0, 100.0, 70.0, [63, 126, 189, 252]
        ),
    }
    for name, result in results.items():
        low, high = result.confidence_95()
        check(
            result.price > 0 and low < result.price < high,
            f"{name} priced at {result.price:.4f}, CI [{low:.4f}, {high:.4f}]",
        )

    # The European control should tighten the Asian estimate without
    # shifting it outside the uncontrolled confidence interval.
    plain, controlled = results["asian"], results["asian+cv"]
    check(
        controlled.std_error < plain.std_error,
        f"control variate cuts std error {plain.std_error:.5f} "
        f"-> {controlled.std_error:.5f}",
    )
    low, high = plain.confidence_95()
    check(low <= controlled.price <= high, "controlled price agrees with plain")

    print("greeks:")
    greeks = engine.greeks_european(model, "call", 100.0)
    # ATM one-year call under these parameters: delta ~0.60, gamma ~0.0155,
    # vega ~38.7 from Black-Scholes.
    check(0.55 < greeks.delta < 0.65, f"delta {greeks.delta:.4f} near 0.60")
    check(0.010 < greeks.gamma < 0.020, f"gamma {greeks.gamma:.4f} near 0.0155")
    check(35.0 < greeks.vega < 42.0, f"vega {greeks.vega:.2f} near 38.7")
    check(greeks.theta > 0, f"theta {greeks.theta:.2f} positive (dV/dT)")

    print("errors surface as ValueError, not PanicException:")
    bad_inputs = {
        "observation index past the path end": lambda: engine.price_autocallable(
            model, 100.0, 2.0, 100.0, 70.0, [999]
        ),
        "unordered observation schedule": lambda: engine.price_autocallable(
            model, 100.0, 2.0, 100.0, 70.0, [126, 63]
        ),
        "empty observation schedule": lambda: engine.price_autocallable(
            model, 100.0, 2.0, 100.0, 70.0, []
        ),
        "negative spot": lambda: fe.GbmModel(
            spot=-100, r=0.03, q=0.0, sigma=0.25, t=1.0
        ),
        "zero volatility is legal": None,
        "zero paths": lambda: fe.McEngine(paths=0).price_european(model, "call", 100.0),
        "unknown option type": lambda: engine.price_european(model, "banana", 100.0),
    }
    for description, call in bad_inputs.items():
        if call is None:
            continue
        try:
            call()
            check(False, f"{description} raised nothing")
        except ValueError:
            check(True, description)
        except BaseException as exc:  # noqa: BLE001 - that is the bug we test for
            check(False, f"{description} raised {type(exc).__name__}")

    # Zero volatility is a degenerate but valid contract, not an error.
    flat = fe.GbmModel(spot=100, r=0.03, q=0.0, sigma=0.0, t=1.0, steps=12)
    deterministic = engine.price_european(flat, "call", 100.0)
    check(
        abs(deterministic.price - (100 * pow(2.718281828459045, 0.03) - 100) * pow(2.718281828459045, -0.03)) < 1e-6,
        f"zero-vol call is its discounted forward intrinsic ({deterministic.price:.6f})",
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
