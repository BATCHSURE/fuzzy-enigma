"""Quote-derived, statically arbitrage-free restricted SSVI surfaces.

The evaluator uses only the standard library. NumPy/SciPy are imported by
calibration, and QuantLib by Black price queries, only when they are needed.
No function in this module contacts LSEG or substitutes missing market data.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

from .analytics import implied_volatility, quantlib
from .calibration import representatives


EPSILON = 1e-6
THETA_MIN, THETA_MAX = 1e-10, 4.0
RHO_BOUND = 0.999
DOMAIN_TOLERANCE = 1e-12
SURFACE_UNITS = {"implied_vol": "decimal annualised", "total_variance": "IV^2 * ACT/365F years",
                 "strike": "index_points", "forward": "index_points", "discount_factor": "dimensionless"}


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{name} is outside its numeric domain")
    return value


def _date(value, name):
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a YYYY-MM-DD date") from exc


def _time(expiry, as_of):
    result = (_date(expiry, "expiry") - _date(as_of, "as_of")).days / 365.0
    if result <= 0:
        raise ValueError("expiry must be after as_of")
    return result


def _hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _hull(points):
    """Monotone-chain convex hull, including degenerate point/line domains."""
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _inside(point, hull):
    if not hull:
        return True
    tolerance = DOMAIN_TOLERANCE * max(1.0, abs(point[0]), abs(point[1]))
    if len(hull) == 1:
        return max(abs(point[j] - hull[0][j]) for j in (0, 1)) <= tolerance
    for left, right in zip(hull, hull[1:] + hull[:1]):
        dx, dy = right[0] - left[0], right[1] - left[1]
        cross = dx * (point[1] - left[1]) - dy * (point[0] - left[0])
        if cross < -tolerance * max(1.0, math.hypot(dx, dy)):
            return False
    if len(hull) == 2:
        return all(min(hull[0][j], hull[1][j]) - tolerance <= point[j]
                   <= max(hull[0][j], hull[1][j]) + tolerance for j in (0, 1))
    return True


def _ssvi(k, theta, rho, eta):
    if eta == 0:
        return theta, 0.0, 0.0, 1.0
    phi = eta / (math.sqrt(theta) * math.sqrt(1 + theta))
    a = phi * k + rho
    b = math.hypot(a, math.sqrt(1 - rho * rho))
    common = rho + a / b
    w = theta / 2 * (1 + rho * phi * k + b)
    w_k = theta * phi / 2 * common
    w_kk = theta * phi * phi / 2 * (1 - rho * rho) / (b * b * b)
    phi_theta = -phi * (1 + 2 * theta) / (2 * theta * (1 + theta))
    w_theta = w / theta + theta / 2 * phi_theta * k * common
    values = (w, w_k, w_kk, w_theta)
    if any(not math.isfinite(value) for value in values) or w <= 0:
        raise ValueError("SSVI evaluation overflowed or produced non-positive total variance")
    return values


class VolSurface:
    """Restricted gamma=1/2 SSVI with bounded maturity/carry support.

    Observed coverage is a convex hull in (log(K/F), ACT/365F time).
    ``allow_wing`` opts into formula wings; it never enables maturity or
    carry extrapolation. Metadata and public node lists are copied.
    """

    def __init__(self, as_of, rho, eta, nodes, curves, coverage=None, *, epsilon=EPSILON, metadata=None):
        self._as_of = _date(as_of, "as_of").isoformat()
        self._rho = _number(rho, "rho")
        self._eta = _number(eta, "eta", nonnegative=True)
        self._epsilon = _number(epsilon, "epsilon", positive=True)
        if self._epsilon >= 2 or abs(self._rho) >= 1:
            raise ValueError("epsilon must be below 2 and abs(rho) below 1")
        if self._eta * (1 + abs(self._rho)) > 2 - self._epsilon:
            raise ValueError("eta violates the sufficient SSVI no-arbitrage constraint")
        self._nodes = []
        for node in nodes:
            expiry = _date(node["expiry"], "node expiry").isoformat()
            t = _time(expiry, self._as_of)
            supplied = _number(node.get("t", t), "node time", positive=True)
            theta = _number(node["theta"], "ATM total variance", positive=True)
            if not math.isclose(t, supplied, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("node time disagrees with date-based ACT/365F")
            if self._nodes and (t <= self._nodes[-1]["t"] or theta < self._nodes[-1]["theta"]):
                raise ValueError("nodes must have strictly increasing expiries and nondecreasing total variance")
            self._nodes.append({"expiry": expiry, "t": t, "theta": theta})
        if not self._nodes:
            raise ValueError("at least one surface node is required")
        self._curves = []
        for curve in curves:
            expiry = _date(curve["expiry"], "curve expiry").isoformat()
            if _date(curve.get("as_of", self._as_of), "curve as_of").isoformat() != self._as_of:
                raise ValueError("curve as_of differs from surface as_of")
            if not isinstance(curve.get("source"), str) or not curve["source"]:
                raise ValueError("curve source must be supplied")
            t = _time(expiry, self._as_of)
            if "t" in curve and not math.isclose(_number(curve["t"], "curve time", positive=True), t,
                                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("curve time disagrees with date-based ACT/365F")
            self._curves.append({**deepcopy(curve), "as_of": self._as_of, "expiry": expiry, "t": t,
                                 "discount_factor": _number(curve["discount_factor"], "discount factor", positive=True),
                                 "forward": _number(curve["forward"], "forward", positive=True)})
        self._curves.sort(key=lambda curve: curve["t"])
        if not self._curves or any(a["t"] == b["t"] for a, b in zip(self._curves, self._curves[1:])):
            raise ValueError("non-empty unambiguous curve nodes are required")
        if self._curves[0]["t"] > self._nodes[0]["t"] or self._curves[-1]["t"] < self._nodes[-1]["t"]:
            raise ValueError("carry curves must cover every surface maturity")
        self._coverage = None if coverage is None else [
            {"k": _number(point["k"], "coverage k"), "t": _number(point["t"], "coverage time", positive=True)}
            for point in coverage]
        if self._coverage is not None and not self._coverage:
            raise ValueError("supplied observed coverage must not be empty")
        if self._coverage is not None and any(not self._nodes[0]["t"] <= p["t"] <= self._nodes[-1]["t"]
                                              for p in self._coverage):
            raise ValueError("coverage times must belong to the fitted maturity range")
        self._hull = _hull([(point["k"], point["t"]) for point in self._coverage]) if self._coverage else []
        self._metadata = deepcopy(metadata or {})
        if "as_of" in self._metadata and self._metadata["as_of"] != self.as_of:
            raise ValueError("metadata as_of differs from surface as_of")
        self._metadata["as_of"] = self.as_of
        if "source" in self._metadata:
            if not isinstance(self._metadata["source"], dict) or self._metadata["source"].get("as_of") != self.as_of:
                raise ValueError("source as_of differs from surface as_of")
        curve_hash = _hash(self._curves)
        if "curve_sha256" in self._metadata and self._metadata["curve_sha256"] != curve_hash:
            raise ValueError("metadata curve hash differs from normalised carry nodes")
        self._metadata["curve_sha256"] = curve_hash
        json.dumps(self._metadata, allow_nan=False)

    as_of = property(lambda self: self._as_of)
    rho = property(lambda self: self._rho)
    eta = property(lambda self: self._eta)
    epsilon = property(lambda self: self._epsilon)
    nodes = property(lambda self: deepcopy(self._nodes))
    curves = property(lambda self: deepcopy(self._curves))
    coverage = property(lambda self: deepcopy(self._coverage))
    metadata = property(lambda self: deepcopy(self._metadata))

    def _theta(self, t):
        t = _number(t, "time", positive=True)
        if not self._nodes[0]["t"] <= t <= self._nodes[-1]["t"]:
            raise ValueError("maturity lies outside fitted surface support; extrapolation is disabled")
        times = [node["t"] for node in self._nodes]
        index = bisect_left(times, t)
        if index < len(times) and t == times[index]:
            left = ((self._nodes[index]["theta"] - self._nodes[index-1]["theta"]) / (t - times[index-1])) if index else None
            right = ((self._nodes[index+1]["theta"] - self._nodes[index]["theta"]) / (times[index+1] - t)) if index+1 < len(times) else None
            return self._nodes[index]["theta"], left, right
        lower, upper = self._nodes[index-1], self._nodes[index]
        slope = (upper["theta"] - lower["theta"]) / (upper["t"] - lower["t"])
        return lower["theta"] + (t - lower["t"]) * slope, slope, slope

    def _check(self, k, t, allow_wing):
        k, t = _number(k, "log-forward moneyness"), _number(t, "time", positive=True)
        theta, left, right = self._theta(t)
        if not isinstance(allow_wing, bool):
            raise ValueError("allow_wing must be Boolean")
        if not allow_wing and not _inside((k, t), self._hull):
            raise ValueError("query lies outside observed moneyness coverage; explicit allow_wing=True is required")
        return k, t, theta, left, right

    def total_variance(self, k, t, allow_wing=False):
        k, _, theta, _, _ = self._check(k, t, allow_wing)
        return _ssvi(k, theta, self.rho, self.eta)[0]

    def derivatives(self, k, t, allow_wing=False, *, time_side="right"):
        if time_side not in ("left", "right"):
            raise ValueError("time_side must be 'left' or 'right'")
        k, t, theta, left, right = self._check(k, t, allow_wing)
        w, w_k, w_kk, w_theta = _ssvi(k, theta, self.rho, self.eta)
        w_left = None if left is None else w_theta * left
        w_right = None if right is None else w_theta * right
        chosen = time_side
        if chosen == "right" and w_right is None:
            chosen = "left"
        elif chosen == "left" and w_left is None:
            chosen = "right"
        w_t = w_right if chosen == "right" else w_left
        return {"w": w, "w_k": w_k, "w_kk": w_kk, "w_t": w_t,
                "w_t_left": w_left, "w_t_right": w_right, "w_theta": w_theta,
                "time_side": chosen if w_t is not None else "unavailable", "theta": theta, "t": t, "k": k,
                "wing_extrapolation": not _inside((k, t), self._hull)}

    def _carry(self, expiry):
        t = _time(expiry, self.as_of)
        if not self._curves[0]["t"] <= t <= self._curves[-1]["t"]:
            raise ValueError("expiry lacks supplied carry coverage; carry extrapolation is disabled")
        times = [curve["t"] for curve in self._curves]
        index = bisect_left(times, t)
        if index < len(times) and times[index] == t:
            row = deepcopy(self._curves[index])
            row["carry_interpolation"] = "exact dated node"
            return row
        lower, upper = self._curves[index-1], self._curves[index]
        alpha = (t - lower["t"]) / (upper["t"] - lower["t"])
        return {"as_of": self.as_of, "expiry": _date(expiry, "expiry").isoformat(), "t": t,
                "discount_factor": math.exp((1-alpha)*math.log(lower["discount_factor"]) + alpha*math.log(upper["discount_factor"])),
                "forward": math.exp((1-alpha)*math.log(lower["forward"]) + alpha*math.log(upper["forward"])),
                "source": [lower["source"], upper["source"]], "carry_interpolation": "log-linear inside supplied range"}

    def implied_vol(self, strike, expiry, allow_wing=False):
        strike = _number(strike, "strike", positive=True)
        curve = self._carry(expiry)
        w = self.total_variance(math.log(strike) - math.log(curve["forward"]), curve["t"], allow_wing)
        return math.sqrt(w / curve["t"])

    def black_price(self, option_type, strike, expiry, allow_wing=False):
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        strike = _number(strike, "strike", positive=True)
        curve = self._carry(expiry)
        sigma = self.implied_vol(strike, expiry, allow_wing)
        # BlackCalculator's ITM parity implementation can lose tiny OTM puts
        # to cancellation. blackFormula evaluates the appropriate tails
        # directly; no negative-price clipping is used here.
        ql = quantlib()
        price = float(ql.blackFormula(ql.Option.Call if option_type == "call" else ql.Option.Put,
                                     strike, curve["forward"], sigma*math.sqrt(curve["t"]),
                                     curve["discount_factor"]))
        if not math.isfinite(price) or price < 0:
            raise ValueError("Black price is non-finite or negative")
        return price

    def to_dict(self):
        from .snapshot import redact
        result = {"schema_version": 1, "model": "SSVI", "parameterisation": "modified_power_gamma_half",
                "as_of": self.as_of, "rho": self.rho, "eta": self.eta, "epsilon": self.epsilon,
                "nodes": self.nodes, "curves": self.curves, "coverage": self.coverage,
                "metadata": redact(self.metadata), "time_convention": "ACT/365F date-based",
                "volatility_unit": "decimal", "premium_unit": "index_points",
                "units": deepcopy(SURFACE_UNITS),
                "interpolation": "linear ATM total variance; log-linear dated discount/forward nodes",
                "extrapolation": "no maturity/carry extrapolation; explicit formula wings only"}
        result["curve_sha256"] = _hash(result["curves"])
        result["surface_sha256"] = _hash(result)
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema_version") != 1 or value.get("model") != "SSVI" or value.get("parameterisation") != "modified_power_gamma_half":
            raise ValueError("unsupported surface schema/model/parameterisation")
        if value.get("volatility_unit") != "decimal" or value.get("premium_unit") != "index_points" or value.get("time_convention") != "ACT/365F date-based":
            raise ValueError("unsupported surface units/time convention")
        if value.get("units") != SURFACE_UNITS or value.get("interpolation") != "linear ATM total variance; log-linear dated discount/forward nodes" or value.get("extrapolation") != "no maturity/carry extrapolation; explicit formula wings only":
            raise ValueError("unsupported surface units/interpolation/extrapolation convention")
        if value.get("surface_sha256") != _hash({key: item for key, item in value.items() if key != "surface_sha256"}):
            raise ValueError("surface content hash is missing or inconsistent")
        if value.get("curve_sha256") != _hash(value["curves"]):
            raise ValueError("surface curve hash is inconsistent")
        surface = cls(value["as_of"], value["rho"], value["eta"], value["nodes"], value["curves"],
                      value.get("coverage"), epsilon=value["epsilon"], metadata=value.get("metadata"))
        if _hash(surface.curves) != value["curve_sha256"]:
            raise ValueError("normalised surface curves differ from the saved curve hash")
        return surface

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        from .snapshot import write_json
        path = Path(path)
        write_json(path, self.to_dict())
        return path

    def diagnostics(self):
        times = sorted({node["t"] for node in self._nodes} | {
            (a["t"] + b["t"]) / 2 for a, b in zip(self._nodes, self._nodes[1:])})
        ks = [-10.0, -5.0] + [-2 + i * .01 for i in range(401)] + [5.0, 10.0]
        minimum_g, minimum_calendar = math.inf, math.inf
        wings = []
        for t in times:
            theta, left, right = self._theta(t)
            phi = self.eta / (math.sqrt(theta) * math.sqrt(1+theta))
            wings.extend([theta*phi*(1-self.rho)/2, theta*phi*(1+self.rho)/2])
            for k in ks:
                w, w_k, w_kk, w_theta = _ssvi(k, theta, self.rho, self.eta)
                g = (1-k*w_k/(2*w))**2 - w_k*w_k/4*(1/w + .25) + w_kk/2
                minimum_g = min(minimum_g, g)
                for slope in (left, right):
                    if slope is not None:
                        minimum_calendar = min(minimum_calendar, w_theta * slope)
        if not math.isfinite(minimum_calendar):
            minimum_calendar = None
        product = self.eta*(1+abs(self.rho))
        return {"static_arbitrage": {
                    "passed": minimum_g >= -1e-10 and (minimum_calendar is None or minimum_calendar >= -1e-10),
                    "analytic_certificate": "restricted SSVI sufficient conditions, Gatheral-Jacquier Corollary4.1/Eq4.5",
                    "min_density_g": minimum_g, "min_calendar_derivative": minimum_calendar,
                    "max_wing_slope": max(wings), "constraint_margins": {
                        "eta_margin": 2-product, "required_epsilon": self.epsilon,
                        "rho_margin": 1-abs(self.rho), "minimum_theta": min(node["theta"] for node in self._nodes),
                        "minimum_theta_increment": min((b["theta"]-a["theta"] for a,b in zip(self._nodes,self._nodes[1:])), default=None)},
                    "scan_grid": {"times": times, "k_min": -10, "k_max": 10, "points_per_time": len(ks),
                                  "meaning": "formula-domain numerical checks supplement the analytic certificate; not quote coverage"}},
                "coverage": {"policy": "observed convex hull" if self._coverage else "unspecified formula domain",
                             "vertices": [{"k": k, "t": t} for k,t in self._hull],
                             "observations": len(self._coverage) if self._coverage else 0,
                             "maturity_range": [self._nodes[0]["t"], self._nodes[-1]["t"]],
                             "carry_range": [self._curves[0]["t"], self._curves[-1]["t"]]},
                "interpolation": "linear theta; time derivatives are one-sided at knots",
                "meaning": "static no-arbitrage and implementation checks do not establish market fit or Local Vol stability"}


def _pava(values, weights):
    """Count-weighted isotonic seed; no quote prices are replaced."""
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index+1, weight*value, float(weight)])
        while len(blocks) > 1 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    result = [0.0] * len(values)
    for start, end, total, weight in blocks:
        result[start:end] = [total/weight] * (end-start)
    return result


def _initialization(train, expiries):
    raw, weights, observations = [], [], []
    for expiry in expiries:
        rows = sorted((row for row in train if row["expiry"] == expiry), key=lambda row: row["strike"])
        lower = [row for row in rows if row["strike"] <= row["forward"]]
        upper = [row for row in rows if row["strike"] >= row["forward"]]
        if lower and upper:
            left, right = lower[-1], upper[0]
            kl = math.log(left["strike"]) - math.log(left["forward"])
            kr = math.log(right["strike"]) - math.log(right["forward"])
            wl, wr = left["iv"]**2*left["t"], right["iv"]**2*right["t"]
            theta = wl if kl == kr else (kr*wl-kl*wr)/(kr-kl)
            used, basis = [left["ric"], right["ric"]], "training bracketing total-variance interpolation at k=0"
        else:
            row = min(rows, key=lambda row: abs(math.log(row["strike"])-math.log(row["forward"])))
            theta = row["iv"]**2*row["t"]
            used, basis = [row["ric"]], "nearest training observation; one-sided ATM coverage"
        raw.append(theta)
        weights.append(len(rows))
        observations.append({"expiry": expiry, "rics": used, "basis": basis, "training_count": len(rows)})
    isotonic = _pava(raw, weights)
    projected = [min(THETA_MAX, max(THETA_MIN, value)) for value in isotonic]
    return {"expiries": expiries, "raw_theta": raw, "isotonic_theta": isotonic,
            "projected_theta": projected, "scales": projected, "weights": weights,
            "observations": observations, "adjustments": [b-a for a,b in zip(raw, projected)],
            "meaning": "training-only initializer; isotonic/bound projection affects seeds, never observed prices"}


def _curves(validated, rows):
    if validated.get("curves"):
        return deepcopy(validated["curves"])
    result = {}
    for row in rows:
        curve = {"as_of": row["as_of"], "expiry": row["expiry"], "discount_factor": row["discount_factor"],
                 "forward": row["forward"], "source": row.get("curve_source", "explicit validated row")}
        if row["expiry"] in result and result[row["expiry"]] != curve:
            raise ValueError("validated rows have inconsistent carry for one expiry")
        result[row["expiry"]] = curve
    return [result[key] for key in sorted(result)]


def _metrics(rows, prices):
    if not rows:
        return {"count": 0, "price_rmse": None, "weighted_rmse": None, "mean_abs_iv_error_bp": None,
                "iv_comparable_count": 0}
    errors = [price-row["close"] for row,price in zip(rows,prices)]
    iv_errors = []
    for row, price in zip(rows, prices):
        try:
            iv_errors.append(abs(implied_volatility(row, price)-row["iv"])*10_000)
        except (ValueError, RuntimeError, TypeError):
            pass
    return {"count": len(rows), "price_rmse": math.sqrt(sum(error*error for error in errors)/len(rows)),
            "weighted_rmse": math.sqrt(sum((error/row["uncertainty_price"])**2 for row,error in zip(rows,errors))/len(rows)),
            "mean_abs_iv_error_bp": sum(iv_errors)/len(iv_errors) if iv_errors else None,
            "iv_comparable_count": len(iv_errors)}


def _prediction(surface, row, role, *, allow_wing=False):
    prediction = {**deepcopy(row), "split": role, "calibration_role": role,
                  "model_price": None, "model_iv": None, "price_error": None, "weighted_error": None,
                  "iv_error_bp": None, "iv_error_reason": None, "query_error": None,
                  "wing_extrapolation": False, "model": "SSVI"}
    try:
        k = math.log(row["strike"])-math.log(row["forward"])
        prediction["wing_extrapolation"] = not _inside((k,row["t"]), surface._hull)
        iv = surface.implied_vol(row["strike"], row["expiry"], allow_wing=allow_wing)
        price = surface.black_price(row["kind"], row["strike"], row["expiry"], allow_wing=allow_wing)
        prediction.update(model_price=price, model_iv=iv, price_error=price-row["close"],
                          weighted_error=(price-row["close"])/row["uncertainty_price"])
        if row.get("iv") is not None:
            prediction["iv_error_bp"] = (iv-row["iv"])*10_000
        else:
            prediction["iv_error_reason"] = row.get("iv_error_reason", "quote IV unavailable under specified carry")
    except (ValueError, RuntimeError, OverflowError) as exc:
        prediction["query_error"] = str(exc)
    return prediction


def _fit_core(validated, *, starts, maxiter, seed, training_expiries=None, representative_rics=None):
    source = deepcopy(validated.get("source", {}))
    as_of = _date(source.get("as_of"), "source.as_of").isoformat()
    rows = deepcopy(validated.get("accepted", []))
    for row in rows:
        row["as_of"] = _date(row.get("as_of", as_of), "row as_of").isoformat()
        if row["as_of"] != as_of:
            raise ValueError("surface fit requires one common as_of date")
        t = _time(row["expiry"], as_of)
        if not math.isclose(_number(row["t"], "row time", positive=True), t, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("row time disagrees with ACT/365F")
        row["t"] = t
        for key in ("strike", "forward", "discount_factor", "uncertainty_price"):
            row[key] = _number(row[key], key, positive=True)
        row["close"] = _number(row["close"], "Close", nonnegative=True)
        if row["kind"] not in ("call", "put") or row["split"] not in ("train", "holdout"):
            raise ValueError("unsupported quote kind/split")
    selected = representatives(rows) if representative_rics is None else [row for row in rows if row["ric"] in representative_rics]
    counts = defaultdict(int)
    for row in selected:
        if row["split"] == "train":
            counts[row["expiry"]] += 1
    all_expiries = sorted({row["expiry"] for row in selected})
    qualifying = [expiry for expiry in all_expiries if counts[expiry] >= 5 and
                  (training_expiries is None or expiry in training_expiries)]
    excluded = [{"expiry": expiry, "training_strikes": counts[expiry],
                 "holdout_strikes": sum(row["expiry"] == expiry and row["split"] == "holdout" for row in selected),
                 "reason": "removed for maturity holdout" if counts[expiry] >= 5 else "fewer than 5 training OTM strike groups"}
                for expiry in all_expiries if expiry not in qualifying]
    train = [row for row in selected if row["expiry"] in qualifying and row["split"] == "train"]
    holdout = [row for row in selected if row["expiry"] in qualifying and row["split"] == "holdout"]
    settings = {"starts": starts, "seed": seed, "maxiter": maxiter, "method": "SLSQP", "loss": "mean soft_l1",
                "ftol": 1e-10, "epsilon": EPSILON, "rho_bounds": [-RHO_BOUND,RHO_BOUND],
                "eta_bounds": [0,2-EPSILON], "theta_bounds": [THETA_MIN,THETA_MAX],
                "theta_units": "ATM total variance, not annual variance", "scaling": "theta/train-only isotonic initializer",
                "split": "existing every-fifth strike groups; call/put share split", "minimum_training_strikes": 5,
                "weighting": "fixed pre-fit uncertainty_price; measured historical spread or assumed Close IV uncertainty",
                "selection": "lowest training loss among converged feasible starts; holdout never selects a start"}
    base = {"schema_version": 1, "model": "SSVI", "source": source, "source_sha256": _hash(source),
            "dataset_sha256": validated.get("dataset_sha256"), "input_sha256": _hash({"source": source,"rows": rows,"curves": validated.get("curves")}),
            "status": "insufficient_data", "surface": None, "parameters": None, "chosen_start": None,
            "train": {"count":len(train)}, "holdout": {"count":len(holdout)}, "predictions": [], "starts": [],
            "settings": settings, "qualifying_expiries": qualifying, "excluded_expiries": excluded,
            "training_groups": [row["group"] for row in train], "holdout_groups": [row["group"] for row in holdout],
            "diagnostics": {}, "expiry_holdouts": [], "sensitivities": []}
    if len(qualifying) < 3:
        return {**base, "reason": "Need at least three expiries with five training OTM strike groups each."}
    for row in train:
        row["iv"] = _number(row.get("iv"), "training quote IV", nonnegative=True)
    try:
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import ndtr
    except ImportError as exc:
        raise RuntimeError("SSVI fitting requires the optional numpy/scipy market dependencies") from exc
    initialization = _initialization(train, qualifying)
    scales = np.array(initialization["scales"])
    indices = np.array([qualifying.index(row["expiry"]) for row in train])
    k = np.array([math.log(row["strike"])-math.log(row["forward"]) for row in train])
    forward, strike, discount, target, uncertainty = [np.array([row[key] for row in train]) for key in
                                                    ("forward","strike","discount_factor","close","uncertainty_price")]
    sign = np.array([1 if row["kind"] == "call" else -1 for row in train])
    n = len(qualifying)

    def objective(vector):
        rho, eta = vector[:2]
        theta = (vector[2:]*scales)[indices]
        c = 1/np.sqrt(theta*(1+theta))
        phi = eta*c
        a = phi*k+rho
        b = np.hypot(a, math.sqrt(max(0.0,1-rho*rho)))
        w = theta/2*(1+rho*phi*k+b)
        sd = np.sqrt(w)
        d1, d2 = -k/sd+sd/2, -k/sd-sd/2
        prices = discount*sign*(forward*ndtr(sign*d1)-strike*ndtr(sign*d2))
        residual = (prices-target)/uncertainty
        if not np.all(np.isfinite(residual)):
            raise ValueError("non-finite surface optimization residual")
        derivative = discount*forward*np.exp(-d1*d1/2)/(math.sqrt(2*math.pi)*2*sd)
        phi_theta = -phi*(1+2*theta)/(2*theta*(1+theta))
        w_theta = w/theta+theta/2*phi_theta*k*(rho+a/b)
        jacobian = np.zeros((len(train),n+2))
        jacobian[:,0] = derivative*theta/2*phi*k*(1+1/b)
        jacobian[:,1] = derivative*theta/2*c*k*(rho+a/b)
        jacobian[np.arange(len(train)),indices+2] = derivative*w_theta*scales[indices]
        robust = np.hypot(1,residual)
        return float(np.mean(robust-1)), np.mean((residual/robust/uncertainty)[:,None]*jacobian,axis=0)

    difference = np.zeros((n-1,n+2))
    for index in range(n-1):
        difference[index,index+2] = -scales[index]/scales[index+1]
        difference[index,index+3] = 1

    def skew_constraint(vector):
        rho, eta = vector[:2]
        return np.array([2-EPSILON-eta*(1+rho), 2-EPSILON-eta*(1-rho)])

    def skew_jacobian(vector):
        rho, eta = vector[:2]
        matrix = np.zeros((2,n+2))
        matrix[0,:2], matrix[1,:2] = (-eta,-1-rho), (eta,-1+rho)
        return matrix

    constraints = [{"type":"ineq","fun":skew_constraint,"jac":skew_jacobian},
                   {"type":"ineq","fun":lambda x:difference@x,"jac":lambda x:difference}]
    bounds = [(-RHO_BOUND,RHO_BOUND),(0,2-EPSILON)]+[(THETA_MIN/scale,THETA_MAX/scale) for scale in scales]
    rng = np.random.default_rng(seed)
    initial = [np.r_[-.7,.5,np.ones(n)]]
    for _ in range(starts-1):
        rho = float(rng.uniform(-.95,.95))
        eta = float(rng.uniform(.1,.9))*(2-EPSILON)/(1+abs(rho))
        low = max(-.5,math.log(THETA_MIN/float(scales.min())))
        high = min(.5,math.log(THETA_MAX/float(scales.max())))
        multiplier = math.exp(float(rng.uniform(low,high))) if low < high else 1.0
        initial.append(np.r_[rho,eta,np.full(n,multiplier)])
    curves = _curves(validated, rows)
    coverage = [{"k":math.log(row["strike"])-math.log(row["forward"]),"t":row["t"]}
                for row in rows if row["expiry"] in qualifying]
    training_hash = _hash([{key:row.get(key) for key in ("ric","group","expiry","t","kind","strike","forward","discount_factor","close","iv","uncertainty_price")} for row in train])
    candidates = []
    import warnings
    for index, vector in enumerate(initial):
        record = {"index":index,"initial":{"rho":float(vector[0]),"eta":float(vector[1]),
                                            "theta_nodes":[float(value) for value in vector[2:]*scales]}}
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always",RuntimeWarning)
                fit = minimize(objective,vector,jac=True,bounds=bounds,constraints=constraints,
                               method="SLSQP",options={"maxiter":maxiter,"ftol":1e-10})
            record.update(success=bool(fit.success),message=str(fit.message),nit=int(fit.nit),
                          nfev=int(fit.nfev),warnings=[str(warning.message) for warning in captured])
            if not np.all(np.isfinite(fit.x)) or not math.isfinite(float(fit.fun)):
                raise ValueError("optimizer returned non-finite parameters or loss")
            rho, eta = map(float,fit.x[:2])
            theta = list(map(float,fit.x[2:]*scales))
            repaired = []
            if abs(rho) > RHO_BOUND or eta < 0 or any(value < THETA_MIN or value > THETA_MAX for value in theta):
                raise ValueError("optimizer violates parameter bounds")
            for j in range(1,n):
                if theta[j] < theta[j-1]:
                    delta = theta[j-1]-theta[j]
                    if delta > 1e-12*max(1e-6,theta[j-1],theta[j]):
                        raise ValueError("optimizer violates monotone total variance")
                    repaired.append({"theta_index":j,"adjustment":delta})
                    theta[j] = theta[j-1]
            excess = eta*(1+abs(rho))-(2-EPSILON)
            if excess > 0:
                if excess > 1e-10:
                    raise ValueError("optimizer violates the eta no-arbitrage constraint")
                replacement = math.nextafter((2-EPSILON)/(1+abs(rho)),0)
                repaired.append({"eta_adjustment":replacement-eta})
                eta = replacement
            nodes = [{"expiry":expiry,"t":_time(expiry,as_of),"theta":value} for expiry,value in zip(qualifying,theta)]
            surface = VolSurface(as_of,rho,eta,nodes,curves,coverage,metadata={"source":source,
                                 "dataset_sha256":base["dataset_sha256"],"input_sha256":base["input_sha256"],
                                 "training_sha256":training_hash,"settings":settings,
                                 "training_groups":base["training_groups"],"holdout_groups":base["holdout_groups"],
                                 "versions":deepcopy(validated.get("versions",{}))})
            if not surface.diagnostics()["static_arbitrage"]["passed"]:
                raise ValueError("independent post-fit static-arbitrage checks failed")
            actual = np.r_[rho,eta,np.array(theta)/scales]
            loss = objective(actual)[0]
            record.update(cost=loss,parameters={"rho":rho,"eta":eta,"theta_nodes":nodes},
                          feasible=True,roundoff_repairs=repaired)
            candidates.append((not bool(fit.success),loss,index,surface))
        except (ValueError,RuntimeError,OverflowError) as exc:
            record.update(feasible=False,error=str(exc))
        base["starts"].append(record)
    if not candidates:
        return {**base,"status":"failed","reason":"No finite, independently feasible start completed.",
                "diagnostics":{"initialization":initialization}}
    incomplete,loss,chosen,surface = min(candidates,key=lambda item:(item[0],item[1],item[2]))
    predictions = [_prediction(surface,row,row["split"] if row["expiry"] in qualifying else "excluded_expiry") for row in rows]
    train_prices = [surface.black_price(row["kind"],row["strike"],row["expiry"]) for row in train]
    holdout_prices = [surface.black_price(row["kind"],row["strike"],row["expiry"]) for row in holdout]
    feasible_records = [record for record in base["starts"] if record.get("feasible")]
    dispersion = {}
    for key in ("rho","eta"):
        values = [record["parameters"][key] for record in feasible_records]
        dispersion[key] = {"min":min(values),"max":max(values),"range":max(values)-min(values)}
    dispersion["theta_nodes"] = []
    for index, node in enumerate(surface.nodes):
        values = [record["parameters"]["theta_nodes"][index]["theta"] for record in feasible_records]
        dispersion["theta_nodes"].append({"expiry":node["expiry"],"t":node["t"],
                                         "min":min(values),"max":max(values),"range":max(values)-min(values)})
    diagnostic = surface.diagnostics()
    diagnostic.update(initialization=initialization,parameter_dispersion=dispersion,
                      parameter_dispersion_meaning="optimizer-start dispersion, not a confidence interval",
                      bound_activity={"rho":abs(surface.rho)>=RHO_BOUND-1e-8,"eta_zero":surface.eta<=1e-8,
                                      "eta_constraint":2-surface.eta*(1+abs(surface.rho))-EPSILON<=1e-8,
                                      "theta_lower":[node["expiry"] for node in surface.nodes if node["theta"]<=THETA_MIN*(1+1e-6)],
                                      "theta_upper":[node["expiry"] for node in surface.nodes if node["theta"]>=THETA_MAX*(1-1e-6)]})
    fit_status = "incomplete" if incomplete else "converged"
    surface = VolSurface(surface.as_of,surface.rho,surface.eta,surface.nodes,surface.curves,surface.coverage,
                         epsilon=surface.epsilon,metadata={**surface.metadata,"fit_status":fit_status,
                                                         "chosen_start":chosen,"training_loss":loss})
    artifact = surface.to_dict()
    return {**base,"status":fit_status,"surface":artifact,
            "parameters":{"rho":surface.rho,"eta":surface.eta,"theta_nodes":surface.nodes},
            "chosen_start":chosen,"training_loss":loss,"training_sha256":training_hash,
            "curve_sha256":artifact["curve_sha256"],"surface_sha256":artifact["surface_sha256"],
            "train":_metrics(train,train_prices),"holdout":_metrics(holdout,holdout_prices),
            "predictions":predictions,"diagnostics":diagnostic,
            "meaning":"quote-derived SSVI market fit; supplier verification and independent code validation remain separate"}


def fit_ssvi(validated, *, starts=8, maxiter=2000, seed=42, diagnostics=True):
    """Fit shared rho/eta and monotone theta nodes from training quotes only.

    Optional diagnostics add independent interior-expiry folds, parallel-rate
    carry scenarios, and fixed Close-IV uncertainty alternatives. None selects
    or changes the base fit. No raw quotes, schedules, or snapshot are mutated.
    """
    if any(isinstance(value,bool) or not isinstance(value,Integral) or value < 1 for value in (starts,maxiter)):
        raise ValueError("starts/maxiter must be positive integers")
    if isinstance(seed,bool) or not isinstance(seed,Integral) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not isinstance(diagnostics,bool):
        raise ValueError("diagnostics must be Boolean")
    starts,maxiter,seed = int(starts),int(maxiter),int(seed)
    result = _fit_core(validated,starts=starts,maxiter=maxiter,seed=seed)
    result["settings"]["diagnostic_runs"] = diagnostics
    if not diagnostics or result["surface"] is None:
        return result
    selected = representatives(validated["accepted"])
    rics = {row["ric"] for row in selected}
    expiries = result["qualifying_expiries"]
    if len(expiries) >= 4:
        for expiry in expiries[1:-1]:
            remaining = [value for value in expiries if value != expiry]
            fold = _fit_core(validated,starts=starts,maxiter=maxiter,seed=seed,
                             training_expiries=remaining,representative_rics=rics)
            targets = [row for row in selected if row["expiry"] == expiry]
            predictions, metrics = [], {"count":0}
            if fold["surface"] is not None:
                surface = VolSurface.from_dict(fold["surface"])
                predictions = [_prediction(surface,row,"expiry_holdout",allow_wing=True) for row in targets]
                usable = [prediction for prediction in predictions if prediction["model_price"] is not None]
                metrics = _metrics(usable,[prediction["model_price"] for prediction in usable])
            result["expiry_holdouts"].append({"held_out_expiry":expiry,"training_expiries":remaining,
                        "status":fold["status"],"parameters":fold["parameters"],"surface":fold["surface"],
                        "chosen_start":fold["chosen_start"],"starts":fold["starts"],"train":fold["train"],
                        "holdout":metrics,"predictions":predictions,"training_groups":fold["training_groups"],
                        "meaning":"whole interior expiry removed, including theta node; formula-wing evaluations explicitly labelled"})
    else:
        result["diagnostics"]["expiry_holdout_unavailable"] = "At least four qualifying expiries are required to leave three for training."
    for variable, shift in (("risk_free",-0.0001),("risk_free",0.0001),
                            ("dividend",-0.0001),("dividend",0.0001)):
        scenario = deepcopy(validated)
        dr, dq = (shift,0.0) if variable == "risk_free" else (0.0,shift)
        entry = {"category":"carry","name":f"{variable}_{'minus' if shift < 0 else 'plus'}_1bp",
                 "rate_shift":dr,"dividend_shift":dq,
                 "policy":("r parallel shift, q fixed: D*=exp(-dr*T), F*=exp(dr*T)" if dr else
                            "q parallel shift, r fixed: D unchanged, F*=exp(-dq*T)")+
                           "; Close, target groups and pre-fit price uncertainties fixed"}
        try:
            invalid_quotes = []
            for row in scenario["accepted"]:
                row["discount_factor"] *= math.exp(-dr*row["t"])
                row["forward"] *= math.exp((dr-dq)*row["t"])
                row["r"] = _number(row["r"],"risk-free rate")+dr
                row["q"] = _number(row["q"],"dividend rate")+dq
                try:
                    row["iv"] = implied_volatility(row,row["close"])
                except (ValueError,RuntimeError) as exc:
                    row["iv"],row["iv_error_reason"] = None,str(exc)
                    invalid_quotes.append({"ric":row["ric"],"reason":str(exc)})
            entry["invalid_quotes"] = invalid_quotes
            scenario["curves"] = [{**curve,"discount_factor":curve["discount_factor"]*math.exp(-dr*_time(curve["expiry"],scenario["source"]["as_of"])),
                                   "forward":curve["forward"]*math.exp((dr-dq)*_time(curve["expiry"],scenario["source"]["as_of"]))}
                                  for curve in _curves(validated,validated["accepted"])]
            if invalid_quotes:
                raise ValueError("carry scenario violates Black price bounds for one or more fixed accepted quotes")
            fit = _fit_core(scenario,starts=starts,maxiter=maxiter,seed=seed,representative_rics=rics)
            entry.update({key:fit.get(key) for key in ("status","parameters","train","holdout","training_groups","holdout_groups","chosen_start")})
        except (ValueError,RuntimeError,OverflowError) as exc:
            entry.update(status="failed",error=str(exc))
        result["sensitivities"].append(entry)
    tick = _number(validated.get("settings",{}).get("tick_floor",.05),"tick floor",positive=True)
    for iv_uncertainty in (.005,.02):
        scenario = deepcopy(validated)
        for row in scenario["accepted"]:
            if not row.get("bid_ask_usable",False):
                row["uncertainty_price"] = max(tick,_number(row.get("vega",0),"quote vega",nonnegative=True)*iv_uncertainty)
        entry = {"category":"weight","name":f"close_iv_uncertainty_{iv_uncertainty:g}",
                 "iv_uncertainty":iv_uncertainty,"policy":"change assumed Close uncertainty only; historical spreads and tick floor fixed"}
        try:
            fit = _fit_core(scenario,starts=starts,maxiter=maxiter,seed=seed,representative_rics=rics)
            entry.update({key:fit.get(key) for key in ("status","parameters","train","holdout","training_groups","holdout_groups","chosen_start")})
        except (ValueError,RuntimeError,OverflowError) as exc:
            entry.update(status="failed",error=str(exc))
        result["sensitivities"].append(entry)
    for entry in result["sensitivities"]:
        if entry.get("train") and entry.get("holdout"):
            entry["comparison"] = {"training_price_rmse_change":entry["train"].get("price_rmse",0)-result["train"]["price_rmse"],
                                   "holdout_price_rmse_change":(entry["holdout"].get("price_rmse")-result["holdout"].get("price_rmse"))
                                   if entry["holdout"].get("price_rmse") is not None and result["holdout"].get("price_rmse") is not None else None}
    return result
