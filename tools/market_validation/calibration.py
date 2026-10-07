"""Global Heston calibration and independent verification of the Rust MC core."""

from __future__ import annotations

from collections import defaultdict
import math
from numbers import Integral

from .analytics import black_price, evaluation_date, implied_volatility, ql_date


PARAMETERS = ("v0", "kappa", "theta", "xi", "rho")
LOWER_BOUNDS = (1e-6, 0.001, 1e-6, 0.0001, -0.999)
UPPER_BOUNDS = (4.0, 20.0, 4.0, 5.0, 0.999)


def representatives(rows):
    """One OTM observation per expiry/strike, stable across call/put duplicates."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    selected = []
    for group in groups.values():
        first = group[0]
        preferred = "call" if first["strike"] >= first["forward"] else "put"
        candidates = [row for row in group if row["kind"] == preferred]
        if candidates:
            selected.append(min(candidates, key=lambda row: row["ric"]))
    return sorted(selected, key=lambda row: (row["expiry"], row["strike"], row["ric"]))


def _parameters(params):
    result = {name: float(params[name]) for name in PARAMETERS}
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("Heston parameters must be finite")
    if result["v0"] < 0 or result["theta"] < 0 or result["kappa"] <= 0 or result["xi"] < 0 or not -1 <= result["rho"] <= 1:
        raise ValueError("Heston parameters are outside the core model domain")
    return result


def _heston_prices(rows, params):
    params = _parameters(params)
    if not rows:
        return []
    dates = {row["as_of"] for row in rows}
    if len(dates) != 1:
        raise ValueError("Heston pricing requires one common valuation date")
    if params["xi"] == 0:
        # The variance ODE is deterministic; its integral replaces sigma² T.
        result = []
        for row in rows:
            t, kappa = row["t"], params["kappa"]
            integrated = params["theta"] * t + (params["v0"] - params["theta"]) * (-math.expm1(-kappa * t)) / kappa
            result.append(black_price(row, math.sqrt(max(integrated / t, 0.0))))
        return result
    with evaluation_date(rows[0]["as_of"]) as ql:
        engines = {}
        result = []
        for row in rows:
            key = (row["expiry"], row["spot"], row["r"], row["q"])
            if key not in engines:
                day_count = ql.Actual365Fixed()
                start = ql_date(row["as_of"])
                risk_free = ql.YieldTermStructureHandle(ql.FlatForward(start, row["r"], day_count))
                dividend = ql.YieldTermStructureHandle(ql.FlatForward(start, row["q"], day_count))
                process = ql.HestonProcess(risk_free, dividend, ql.QuoteHandle(ql.SimpleQuote(row["spot"])),
                                          params["v0"], params["kappa"], params["theta"], params["xi"], params["rho"])
                engines[key] = ql.AnalyticHestonEngine(ql.HestonModel(process), 144)
            payoff = ql.PlainVanillaPayoff(ql.Option.Call if row["kind"] == "call" else ql.Option.Put, row["strike"])
            option = ql.VanillaOption(payoff, ql.EuropeanExercise(ql_date(row["expiry"])))
            option.setPricingEngine(engines[key])
            price = float(option.NPV())
            if not math.isfinite(price) or price < 0:
                raise ValueError("QuantLib returned a non-finite or negative Heston price")
            result.append(price)
        return result


def heston_price(row, params):
    """Independent deterministic price with the row's effective flat carry."""
    return _heston_prices([row], params)[0]


def _metrics(rows, prices):
    if not rows:
        return {"count": 0, "price_rmse": None, "weighted_rmse": None, "mean_abs_iv_error_bp": None}
    errors = [price - row["close"] for row, price in zip(rows, prices)]
    iv_errors = []
    for row, price in zip(rows, prices):
        try:
            iv_errors.append(abs(implied_volatility(row, price) - row["iv"]) * 10_000)
        except (ValueError, RuntimeError):
            pass
    return {"count": len(rows), "price_rmse": math.sqrt(sum(error * error for error in errors) / len(rows)),
            "weighted_rmse": math.sqrt(sum((error / row["uncertainty_price"]) ** 2 for row, error in zip(rows, errors)) / len(rows)),
            "mean_abs_iv_error_bp": sum(iv_errors) / len(iv_errors) if iv_errors else None,
            "iv_comparable_count": len(iv_errors)}


def calibrate(validated, starts=8, max_nfev=500, seed=42):
    """Fit one five-parameter model across expiries using only training groups.

    Each row's price uncertainty is fixed before optimization. Eight seeded
    starts use bounded TRF/soft_l1; selecting a start never consults holdout
    losses, the vendor surface, or a fitted forward from held-out quotes.
    """
    if starts < 1 or max_nfev < 1:
        raise ValueError("starts and max_nfev must be positive")
    selected = representatives(validated["accepted"])
    all_train = [row for row in selected if row["split"] == "train"]
    all_holdout = [row for row in selected if row["split"] == "holdout"]
    counts = defaultdict(int)
    for row in all_train:
        counts[row["expiry"]] += 1
    all_expiries = sorted({row["expiry"] for row in selected})
    qualifying = {expiry for expiry in all_expiries if counts[expiry] >= 5}
    excluded = [{"expiry": expiry, "training_strikes": counts[expiry],
                 "holdout_strikes": sum(row["expiry"] == expiry for row in all_holdout),
                 "reason": "fewer than 5 training OTM strike groups after holdout"}
                for expiry in all_expiries if expiry not in qualifying]
    train = [row for row in all_train if row["expiry"] in qualifying]
    holdout = [row for row in all_holdout if row["expiry"] in qualifying]
    insufficient = len(qualifying) < 3
    base = {"schema_version": 1, "source": validated["source"], "parameters": None, "params": None, "train_counts": dict(counts),
            "selected_train_counts": {expiry: counts[expiry] for expiry in sorted(qualifying)},
            "qualifying_expiries": sorted(qualifying), "excluded_expiries": excluded,
            "train": {"count": len(train)}, "holdout": {"count": len(holdout)}, "starts": [], "predictions": [],
            "settings": {"starts": starts, "max_nfev": max_nfev, "seed": seed, "method": "trf", "loss": "soft_l1",
                         "bounds": {name: [low, high] for name, low, high in zip(PARAMETERS, LOWER_BOUNDS, UPPER_BOUNDS)},
                         "split": "every fifth unique strike per expiry; call/put share group",
                         "weighting": "fixed price uncertainty; historical half-spread with tick floor, or vega times assumed Close IV uncertainty with tick floor"}}
    if insufficient:
        return {**base, "status": "insufficient_data", "reason": "Need at least 3 expiries, each with 5 training OTM strike groups after holdout."}
    try:
        import numpy as np
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("Calibration requires the optional numpy and scipy dependencies.") from exc
    rng = np.random.default_rng(seed)
    variance = max(1e-5, min(2.0, float(np.median([row["iv"] ** 2 for row in train]))))
    initial = [np.array((variance, 1.5, variance, 0.4, -0.7))]
    for _ in range(starts - 1):
        initial.append(np.array((
            np.exp(rng.uniform(np.log(1e-4), np.log(1.0))),
            np.exp(rng.uniform(np.log(0.05), np.log(10.0))),
            np.exp(rng.uniform(np.log(1e-4), np.log(1.0))),
            np.exp(rng.uniform(np.log(0.02), np.log(2.0))),
            rng.uniform(-0.95, 0.95),
        )))
    target = np.array([row["close"] for row in train])
    uncertainty = np.array([row["uncertainty_price"] for row in train])

    def residual(vector):
        params = dict(zip(PARAMETERS, map(float, vector)))
        return (np.array(_heston_prices(train, params)) - target) / uncertainty

    fits = []
    for index, vector in enumerate(initial):
        start_record = {"index": index, "initial": dict(zip(PARAMETERS, map(float, vector)))}
        try:
            fit = least_squares(residual, vector, bounds=(LOWER_BOUNDS, UPPER_BOUNDS), method="trf",
                                loss="soft_l1", x_scale="jac", max_nfev=max_nfev)
            params = dict(zip(PARAMETERS, map(float, fit.x)))
            if not math.isfinite(float(fit.cost)):
                raise ValueError("non-finite training objective")
            start_record.update(params=params, cost=float(fit.cost), success=bool(fit.success),
                                nfev=int(fit.nfev), message=str(fit.message))
            fits.append((float(fit.cost), index, params, bool(fit.success)))
        except (ValueError, RuntimeError, OverflowError) as exc:
            start_record.update(success=False, error=str(exc))
        base["starts"].append(start_record)
    if not fits:
        return {**base, "status": "failed", "reason": "No finite calibration start completed."}
    _, chosen, params, converged = min(fits, key=lambda fit: (fit[0], fit[1]))
    all_rows = validated["accepted"]
    prices = _heston_prices(all_rows, params)
    predictions = []
    for row, price in zip(all_rows, prices):
        role = row["split"] if row["expiry"] in qualifying else "excluded_expiry"
        prediction = {**row, "split": role, "calibration_role": role,
                      "model_price": price, "price_error": price - row["close"],
                      "weighted_error": (price - row["close"]) / row["uncertainty_price"], "model_iv": None,
                      "iv_error_bp": None, "iv_error_reason": None}
        try:
            prediction["model_iv"] = implied_volatility(row, price)
            prediction["iv_error_bp"] = (prediction["model_iv"] - row["iv"]) * 10_000
        except (ValueError, RuntimeError) as exc:
            prediction["iv_error_reason"] = str(exc)
        if row["bid_ask_usable"]:
            prediction["inside_historical_bid_ask"] = row["bid"] <= price <= row["ask"]
        predictions.append(prediction)
    return {**base, "status": "converged" if converged else "incomplete", "parameters": params, "params": params,
            "chosen_start": chosen, "train": _metrics(train, _heston_prices(train, params)),
            "holdout": _metrics(holdout, _heston_prices(holdout, params)), "predictions": predictions,
            "feller_margin": 2 * params["kappa"] * params["theta"] - params["xi"] ** 2,
            "meaning": "Market Close fit under Heston; this is separate from independent core-code verification."}


def _stratified(rows):
    selected = representatives(rows)
    expiries = sorted({row["expiry"] for row in selected})
    if len(expiries) > 3:
        expiries = [expiries[0], expiries[len(expiries) // 2], expiries[-1]]
    result = []
    for expiry in expiries:
        group = [row for row in selected if row["expiry"] == expiry]
        if len(group) > 3:
            group = [group[0], group[len(group) // 2], group[-1]]
        result.extend(group)
    return result


def numerical_validation(validated, params=None, paths=100_000, steps=(64, 256, 1024), seeds=(42, 43, 44)):
    """Compare core MC to independent QL references, never to inverted Close.

    GBM uses fixed sigma=0.20. Heston uses the supplied calibrated parameters.
    Grid changes are reported separately from sample confidence and market fit.
    """
    if paths < 1 or not steps or any(step < 1 for step in steps):
        raise ValueError("positive paths/steps and at least one seed are required")
    try:
        seeds = tuple(seeds)
    except TypeError as exc:
        raise ValueError("seeds must be a non-empty sequence of distinct u64 integers") from exc
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, Integral)
                        or not 0 <= seed <= (1 << 64) - 1 for seed in seeds):
        raise ValueError("seeds must be a non-empty sequence of distinct u64 integers")
    seeds = tuple(map(int, seeds))
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct; repeated RNG streams do not provide independent samples")
    try:
        import fuzzy_enigma as fe
    except ImportError as exc:
        raise RuntimeError("Numerical verification requires the built fuzzy_enigma extension.") from exc
    steps = sorted(set(map(int, steps)))
    rows = _stratified(validated["accepted"])
    params = _parameters(params) if params is not None else None
    records, summaries = [], []
    for row in rows:
        references = {"GBM": black_price(row, 0.2)}
        if params is not None:
            references["Heston"] = heston_price(row, params)
        for name, reference in references.items():
            try:
                reference_iv = implied_volatility(row, reference)
            except (ValueError, RuntimeError):
                reference_iv = None
            local = []
            for step in steps:
                for seed in seeds:
                    engine = fe.McEngine(paths=paths, seed=seed, antithetic=True, parallel=True)
                    if name == "GBM":
                        model = fe.GbmModel(row["spot"], row["r"], row["q"], 0.2, row["t"], steps=step)
                    else:
                        model = fe.HestonModel(row["spot"], row["r"], row["q"], params["v0"], params["kappa"],
                                               params["theta"], params["xi"], params["rho"], row["t"], steps=step)
                    result = engine.price_european(model, row["kind"], row["strike"])
                    gap = result.price - reference
                    tolerance = 4.0 * result.std_error + 1e-7
                    record = {"ric": row["ric"], "expiry": row["expiry"], "strike": row["strike"], "kind": row["kind"],
                              "model": name, "paths": paths, "samples": result.samples, "steps": step, "seed": seed,
                              "reference_price": reference, "mc_price": result.price, "std_error": result.std_error,
                              "price_error": gap, "z_score": gap / result.std_error if result.std_error > 0 else None,
                              "within_sampling_tolerance": abs(gap) <= tolerance, "reference_iv": reference_iv,
                              "mc_iv": None, "iv_error_bp": None, "iv_error_reason": None}
                    try:
                        record["mc_iv"] = implied_volatility(row, result.price)
                        if reference_iv is not None:
                            record["iv_error_bp"] = (record["mc_iv"] - reference_iv) * 10_000
                    except (ValueError, RuntimeError) as exc:
                        record["iv_error_reason"] = str(exc)
                    if name == "GBM" and hasattr(fe, "black_scholes_price"):
                        analytic = fe.black_scholes_price(model, row["kind"], row["strike"])
                        record["core_analytic_price"] = analytic
                        record["core_analytic_error"] = analytic - reference
                    records.append(record)
                    local.append(record)
            finest = [record for record in local if record["steps"] == steps[-1]]
            mean = sum(record["mc_price"] for record in finest) / len(finest)
            se = math.sqrt(sum(record["std_error"] ** 2 for record in finest)) / len(finest)
            fine_gap = mean - reference
            previous = [record for record in local if len(steps) > 1 and record["steps"] == steps[-2]]
            previous_mean = sum(record["mc_price"] for record in previous) / len(previous) if previous else None
            classification = "within_sampling_error" if abs(fine_gap) <= 4 * se + 1e-7 else "unresolved_discretization_or_implementation_error"
            if se == 0 and mean == 0 and reference > 1e-7:
                classification = "insufficient_tail_sampling"
            summaries.append({"ric": row["ric"], "model": name, "finest_steps": steps[-1], "mean_price": mean,
                              "combined_std_error": se, "reference_price": reference, "price_error": fine_gap,
                              "last_grid_change": mean - previous_mean if previous_mean is not None else None,
                              "classification": classification,
                              "meaning": "Core/reference comparison; no conclusion about market-price fit"})
    return {"schema_version": 1, "source": validated["source"], "settings": {"paths": paths, "steps": steps,
            "seeds": list(seeds), "gbm_sigma": 0.2, "heston_params": params}, "records": records,
            "summaries": summaries, "status": "completed" if rows else "no_eligible_quotes"}
