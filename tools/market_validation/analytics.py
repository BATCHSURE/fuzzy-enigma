"""Independent Black analytics and strict, replayable EOD quote validation.

Market packages are imported only when needed. No function in this module
contacts a data service, changes a snapshot, or substitutes missing curves.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import date
import math
import threading


_QL_LOCK = threading.RLock()


def quantlib():
    try:
        import QuantLib as ql
    except ImportError as exc:
        raise RuntimeError("Market analytics requires the optional QuantLib dependency.") from exc
    return ql


def _finite(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} is outside its finite numeric domain")
    return value


def _date(value, name):
    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a YYYY-MM-DD date") from exc


def ql_date(value):
    value = _date(value, "date") if not isinstance(value, date) else value
    return quantlib().Date(value.day, value.month, value.year)


@contextmanager
def evaluation_date(as_of):
    """Restore QuantLib's process-wide evaluation date, including on error."""
    ql = quantlib()
    with _QL_LOCK:
        settings = ql.Settings.instance()
        previous = settings.evaluationDate
        settings.evaluationDate = ql_date(as_of)
        try:
            yield ql
        finally:
            settings.evaluationDate = previous


def _payoff(row):
    ql = quantlib()
    return ql.PlainVanillaPayoff(ql.Option.Call if row["kind"] == "call" else ql.Option.Put, row["strike"])


def black_price(row, sigma):
    """Black price in index points using supplied forward and discount factor."""
    sigma = _finite(sigma, "sigma", nonnegative=True)
    calculator = quantlib().BlackCalculator(
        _payoff(row), row["forward"], sigma * math.sqrt(row["t"]), row["discount_factor"]
    )
    return float(calculator.value())


def black_vega(row, sigma):
    if sigma == 0:
        return 0.0
    return float(quantlib().BlackCalculator(
        _payoff(row), row["forward"], sigma * math.sqrt(row["t"]), row["discount_factor"]
    ).vega(row["t"]))


def price_bounds(row):
    forward, strike, discount = row["forward"], row["strike"], row["discount_factor"]
    if row["kind"] == "call":
        return discount * max(forward - strike, 0.0), discount * forward
    return discount * max(strike - forward, 0.0), discount * strike


def implied_volatility(row, price):
    """Invert Black with a bracketed solver; never clip a supplied price."""
    price = _finite(price, "price", nonnegative=True)
    lower, upper = price_bounds(row)
    if price < lower or price >= upper:
        raise ValueError("price is outside the finite-IV Black bounds")
    if price == lower:
        return 0.0
    try:
        from scipy.optimize import brentq
    except ImportError as exc:
        raise RuntimeError("Implied volatility requires the optional scipy dependency.") from exc
    high = 1.0
    while black_price(row, high) < price and high < 128.0:
        high *= 2.0
    if black_price(row, high) < price:
        raise ValueError("could not bracket a finite implied volatility")
    return float(brentq(lambda sigma: black_price(row, sigma) - price, 0.0, high, xtol=1e-12))


def _spread(row, as_of, diagnostics):
    """Use only explicitly dated historical bid/ask, never live replacements."""
    if row.get("bid") is None and row.get("ask") is None:
        return None
    bid_date = row.get("bid_date", row.get("bid_ask_date"))
    ask_date = row.get("ask_date", row.get("bid_ask_date"))
    try:
        bid = _finite(row.get("bid"), "bid", nonnegative=True)
        ask = _finite(row.get("ask"), "ask", nonnegative=True)
        if _date(bid_date, "bid_date") != as_of or _date(ask_date, "ask_date") != as_of:
            raise ValueError("bid/ask dates do not match the Close date")
        if ask < bid:
            raise ValueError("bid exceeds ask")
    except ValueError as exc:
        diagnostics.append({"code": "bid_ask_unavailable", "ric": row.get("ric"), "reason": str(exc)})
        return None
    row["bid"], row["ask"] = bid, ask
    return (ask - bid) / 2.0


def _static_arbitrage(rows, tick_floor):
    diagnostics = []
    groups = defaultdict(dict)
    for row in rows:
        groups[row["expiry"]].setdefault(row["strike"], {}).setdefault(row["kind"], row)
    for expiry, strikes in sorted(groups.items()):
        for strike, kinds in sorted(strikes.items()):
            if "call" in kinds and "put" in kinds:
                call, put = kinds["call"], kinds["put"]
                target = call["discount_factor"] * (call["forward"] - strike)
                gap = call["close"] - put["close"] - target
                tolerance = max(tick_floor, call["uncertainty_price"] + put["uncertainty_price"])
                if abs(gap) > tolerance:
                    diagnostics.append({"code": "close_put_call_parity", "expiry": expiry, "strike": strike,
                                        "gap": gap, "tolerance": tolerance,
                                        "meaning": "Close consistency diagnostic; not an executable arbitrage claim"})
                if call["bid_ask_usable"] and put["bid_ask_usable"]:
                    if not call["bid"] - put["ask"] - tick_floor <= target <= call["ask"] - put["bid"] + tick_floor:
                        diagnostics.append({"code": "bid_ask_put_call_parity", "expiry": expiry, "strike": strike})
        for kind in ("call", "put"):
            selected = [kinds[kind] for _, kinds in sorted(strikes.items()) if kind in kinds]
            slopes = []
            for left, right in zip(selected, selected[1:]):
                dk = right["strike"] - left["strike"]
                difference = right["close"] - left["close"]
                low, high = (-left["discount_factor"] * dk, 0.0) if kind == "call" else (0.0, left["discount_factor"] * dk)
                if difference < low - tick_floor or difference > high + tick_floor:
                    diagnostics.append({"code": "close_vertical_spread", "expiry": expiry, "kind": kind,
                                        "rics": [left["ric"], right["ric"]], "price_difference": difference,
                                        "meaning": "Close consistency diagnostic; historical bid/ask feasibility is not established"})
                slopes.append((difference / dk, dk, left, right))
            for left_slope, right_slope in zip(slopes, slopes[1:]):
                if left_slope[0] > right_slope[0] + tick_floor / min(left_slope[1], right_slope[1]):
                    diagnostics.append({"code": "close_convexity", "expiry": expiry, "kind": kind,
                                        "rics": [left_slope[2]["ric"], left_slope[3]["ric"], right_slope[3]["ric"]],
                                        "meaning": "Close consistency diagnostic; not an executable arbitrage claim"})
    return diagnostics


def _vendor_surface(snapshot, as_of, diagnostics):
    surface = snapshot.get("vendor_surface")
    if not surface:
        diagnostics.append({"code": "vendor_surface_unavailable"})
        return None
    if not isinstance(surface, dict) or surface.get("as_of") != as_of.isoformat() or surface.get("volatility_unit") != "decimal" or str(surface.get("convention", "")).lower() != "black":
        diagnostics.append({"code": "vendor_surface_incompatible", "reason": "same-date Black decimal-vol metadata required"})
        return None
    if surface.get("metadata_basis") == "SDK_example_convention" or surface.get("unit_assumption") or surface.get("convention_assumption"):
        diagnostics.append({"code": "vendor_surface_unverified", "reason": "legacy supplier unit or convention assumptions require fresh typed evidence"})
        return None
    points = []
    for point in surface.get("points", []):
        try:
            points.append({**point, "expiry": _date(point["expiry"], "vendor expiry").isoformat(),
                           "strike": _finite(point["strike"], "vendor strike", positive=True),
                           "iv": _finite(point["iv"], "vendor IV", nonnegative=True)})
        except (KeyError, TypeError, ValueError):
            diagnostics.append({"code": "vendor_surface_invalid_point"})
    return {**surface, "points": points} if points else None


def validate_snapshot(snapshot, *, min_days=7, max_days=365, min_moneyness=0.8,
                      max_moneyness=1.2, iv_uncertainty=0.01, tick_floor=0.05):
    """Validate explicit-date EOD observations and assign grouped holdout data.

    Date-based ACT/365F is shared by core MC and QuantLib. Close observations
    are never called mid prices. Diagnostics retain suspect Close data rather
    than silently replacing prices or manufacturing historical bid/ask.
    """
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema_version")
    source = dict(snapshot.get("source", {}))
    as_of = _date(source.get("as_of"), "source.as_of")
    if source.get("price_basis") != "close":
        raise ValueError("source.price_basis must explicitly be 'close'")
    spot = _finite(snapshot.get("spot"), "spot", positive=True)
    iv_uncertainty = _finite(iv_uncertainty, "iv_uncertainty", positive=True)
    tick_floor = _finite(tick_floor, "tick_floor", positive=True)
    if not 0 < min_days <= max_days or not 0 < min_moneyness < max_moneyness:
        raise ValueError("invalid maturity or moneyness filter")
    diagnostics, accepted, rejected = [], [], []
    curves = {}
    ambiguous_curves = set()
    for curve in snapshot.get("curves", []):
        try:
            expiry = _date(curve["expiry"], "curve.expiry").isoformat()
            if _date(curve["as_of"], "curve.as_of") != as_of:
                raise ValueError("curve as_of does not match quotes")
            if not curve.get("source"):
                raise ValueError("curve source is required")
            discount = _finite(curve["discount_factor"], "discount_factor", positive=True)
            forward = _finite(curve["forward"], "forward", positive=True)
            if expiry in curves or expiry in ambiguous_curves:
                ambiguous_curves.add(expiry)
                curves.pop(expiry, None)
                raise ValueError("duplicate curve expiry")
            curves[expiry] = {**curve, "discount_factor": discount, "forward": forward}
        except (KeyError, TypeError, ValueError) as exc:
            diagnostics.append({"code": "invalid_curve", "expiry": curve.get("expiry"), "reason": str(exc)})
    seen_rics = set()
    for quote in snapshot.get("quotes", []):
        row = dict(quote)
        ric = row.get("ric")
        try:
            if not isinstance(ric, str) or not ric:
                raise ValueError("RIC is required")
            if ric in seen_rics:
                raise ValueError("duplicate RIC")
            seen_rics.add(ric)
            expiry_date = _date(row["expiry"], "expiry")
            days = (expiry_date - as_of).days
            if not min_days <= days <= max_days:
                raise ValueError(f"expiry must be {min_days}-{max_days} calendar days after as_of")
            if _date(row["close_date"], "close_date") != as_of:
                raise ValueError("Close observation date does not match snapshot as_of")
            for key, expected in (("exercise_style", "european"), ("settlement", "pm"), ("currency", "usd"), ("premium_unit", "index_points")):
                if str(row.get(key, "")).lower() != expected:
                    raise ValueError(f"{key} must be {expected}")
            kind = str(row.get("kind", "")).lower()
            if kind not in ("call", "put"):
                raise ValueError("kind must be call or put")
            strike = _finite(row["strike"], "strike", positive=True)
            close = _finite(row["close"], "close", nonnegative=True)
            expiry = expiry_date.isoformat()
            if expiry not in curves:
                raise ValueError("no explicit same-as_of discount factor and forward for expiry")
            curve = curves[expiry]
            forward, discount, t = curve["forward"], curve["discount_factor"], days / 365.0
            if not min_moneyness <= strike / forward <= max_moneyness:
                raise ValueError("strike/forward is outside the moneyness window")
            r = -math.log(discount) / t
            q = r - (math.log(forward) - math.log(spot)) / t
            row.update(expiry=expiry, kind=kind, strike=strike, close=close, t=t, r=r, q=q,
                       spot=spot, forward=forward, discount_factor=discount, as_of=as_of.isoformat(),
                       curve_source=curve["source"], reference_kind="close", group=f"{expiry}:{strike!r}")
            row["iv"] = implied_volatility(row, close)
            row["vega"] = black_vega(row, row["iv"])
            half_spread = _spread(row, as_of, diagnostics)
            row["bid_ask_usable"] = half_spread is not None
            row["uncertainty_price"] = max(tick_floor, half_spread if half_spread is not None else row["vega"] * iv_uncertainty)
            row["uncertainty_source"] = "historical_spread_with_tick_floor" if half_spread is not None else "assumed_close_iv_uncertainty"
            accepted.append(row)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            rejected.append({"ric": ric, "reason": str(exc)})
    expiry_groups = defaultdict(set)
    for row in accepted:
        expiry_groups[row["expiry"]].add(row["strike"])
    splits = {}
    for expiry, strikes in expiry_groups.items():
        for index, strike in enumerate(sorted(strikes), start=1):
            splits[(expiry, strike)] = "holdout" if index % 5 == 0 else "train"
    accepted.sort(key=lambda row: (row["expiry"], row["strike"], row["kind"], row["ric"]))
    for row in accepted:
        row["split"] = splits[(row["expiry"], row["strike"])]
    diagnostics.extend(_static_arbitrage(accepted, tick_floor))
    if any(not row["bid_ask_usable"] for row in accepted):
        diagnostics.append({"code": "historical_bid_ask_missing", "meaning": "No tradable interval or spread-pass conclusion for those Close observations"})
    diagnostics.append({"code": "time_convention", "value": "ACT/365F date-based EOD; exact expiry timestamp metadata is preserved"})
    return {"schema_version": 1, "accepted": accepted, "rejected": rejected, "diagnostics": diagnostics,
            "capture_diagnostics": snapshot.get("diagnostics", {}),
            "source": source, "spot": spot, "curves": list(curves.values()),
            "vendor_surface": _vendor_surface(snapshot, as_of, diagnostics),
            "settings": {"min_days": min_days, "max_days": max_days, "min_moneyness": min_moneyness,
                         "max_moneyness": max_moneyness, "iv_uncertainty": iv_uncertainty, "tick_floor": tick_floor}}
