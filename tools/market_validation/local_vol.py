"""Explicit smooth SSVI exports and an independent backward-PDE reference.

The fitted display/calibration surface is never modified. This module creates
a separate, hashed pricing representation with a C1 monotone ATM variance
curve and a declared finite short-time extension. NumPy/SciPy are needed only
by the finite-difference reference, never by the artifact evaluator.
"""

from bisect import bisect_right
from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from .surface import VolSurface, _hash, _inside, _number, _hull


POLICIES = {
    "interpolation": "pchip_theta_c1",
    "short_time": "global_shifted_phi",
    "wings": "analytic_ssvi_all_real_k",
    "time_domain": "bounded_no_long_extrapolation",
}
ARTIFACT_FIELDS = {
    "schema_version", "model", "as_of", "currency", "asset", "spot", "rho", "eta",
    "theta_floor", "epsilon", "min_denominator", "nodes", "coefficients", "curves",
    "coverage", "source_surface_sha256", "source_dataset_sha256", "source_curve_sha256",
    "fit_status", "source_fit_status", "artifact_sha256", *POLICIES,
}
DEFAULT_MIN_DENOMINATOR = 1e-10
REFERENCE_TOLERANCES = {
    "python_rust_atol": 1e-12, "python_rust_rtol": 1e-10,
    "first_derivative_rtol": 1e-6, "second_derivative_rtol": 1e-4,
    "flat_pde_price_abs_spot": 5e-5,
    "market_pde_price_abs_spot": 5e-4,
    "market_pde_refinement_abs_spot": 5e-4,
    "mc_sigma_multiple": 4.0,
    "mc_discretisation_abs_spot": 2e-4,
}


def _iso(value, name):
    if isinstance(value, datetime):
        raise ValueError(f"{name} must be a date without a time")
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError(f"{name} must be a canonical YYYY-MM-DD date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical YYYY-MM-DD date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must be a canonical YYYY-MM-DD date")
    return value


def _bits(value):
    return struct.pack(">d", float(value)).hex()


def artifact_hash(value):
    """Rust-compatible SHA256 over length-delimited strings and f64 bits."""
    fields = ["local_vol_artifact_v1", value["as_of"], value["currency"], value["asset"]]
    fields += [_bits(value[name]) for name in
               ("spot", "rho", "eta", "theta_floor", "epsilon", "min_denominator")]
    fields += [value[name] for name in
               ("source_surface_sha256", "source_dataset_sha256", "source_curve_sha256")]
    fields += [value[name] for name in POLICIES]
    fields += [value["fit_status"], value["source_fit_status"], str(len(value["nodes"]))]
    for node in value["nodes"]:
        fields += [_bits(node[name]) for name in ("t", "theta", "tangent")]
    fields.append(str(len(value["coefficients"])))
    for row in value["coefficients"]:
        fields += [_bits(row[name]) for name in ("a", "b", "c", "d")]
    fields.append(str(len(value["curves"])))
    for row in value["curves"]:
        fields += [row["date"], _bits(row["discount_factor"]), _bits(row["forward"]), row["source"]]
    fields.append(str(len(value["coverage"])))
    for row in value["coverage"]:
        fields += [_bits(row[name]) for name in ("k", "t")]
    digest = hashlib.sha256()
    for item in fields:
        payload = item.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _pchip(times, values):
    """Fritsch-Butland PCHIP slopes, including standard one-sided endpoints."""
    h = [b-a for a, b in zip(times, times[1:])]
    secants = [(b-a)/step for a, b, step in zip(values, values[1:], h)]
    if len(times) == 2:
        slopes = [secants[0], secants[0]]
    else:
        slopes = [0.0]*len(times)
        for i in range(1, len(times)-1):
            left, right = secants[i-1], secants[i]
            if left > 0 and right > 0:
                w1, w2 = 2*h[i]+h[i-1], h[i]+2*h[i-1]
                slopes[i] = (w1+w2)/(w1/left+w2/right)

        def endpoint(h0, h1, d0, d1):
            result = ((2*h0+h1)*d0-h0*d1)/(h0+h1)
            if result*d0 <= 0:
                return 0.0
            if d0*d1 <= 0 and abs(result) > 3*abs(d0):
                return 3*d0
            return result

        slopes[0] = endpoint(h[0], h[1], secants[0], secants[1])
        slopes[-1] = endpoint(h[-1], h[-2], secants[-1], secants[-2])
    coefficients = []
    for i, step in enumerate(h):
        delta = values[i+1]-values[i]
        coefficients.append({"a": values[i], "b": step*slopes[i],
                             "c": 3*delta-step*(2*slopes[i]+slopes[i+1]),
                             "d": -2*delta+step*(slopes[i]+slopes[i+1])})
    return slopes, coefficients


def _regularized(k, theta, rho, eta, floor):
    phi = eta/(math.sqrt(theta+floor)*math.sqrt(1+theta))
    a = phi*k+rho
    b = math.hypot(a, math.sqrt(1-rho*rho))
    common = rho+a/b
    factor = (1+rho*phi*k+b)/2
    factor_k = phi*common/2
    factor_kk = phi*phi*(1-rho*rho)/(2*b*b*b)
    phi_theta = -phi/2*(1/(theta+floor)+1/(1+theta))
    w_theta = factor+theta*phi_theta*k*common/2
    return theta*factor, theta*factor_k, theta*factor_kk, w_theta, factor, factor_k, factor_kk


class SmoothLocalVolSurface:
    """Validated pricing-only surface; all real k, bounded [0, last tenor]."""

    def __init__(self, artifact, *, export_report=None):
        value = deepcopy(artifact)
        if not isinstance(value, dict) or set(value) != ARTIFACT_FIELDS:
            raise ValueError("local-vol artifact fields are missing or unknown")
        if (isinstance(value["schema_version"], bool)
                or not isinstance(value["schema_version"], int)
                or value["schema_version"] != 1
                or value["model"] != "regularized_ssvi_local_vol"):
            raise ValueError("unsupported local-vol artifact schema/model")
        if any(value[name] != policy for name, policy in POLICIES.items()):
            raise ValueError("unsupported local-vol interpolation/extension policy")
        value["as_of"] = _iso(value["as_of"], "as_of")
        if not isinstance(value["currency"], str) or len(value["currency"]) != 3 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in value["currency"]):
            raise ValueError("currency must be an explicit three-letter uppercase code")
        if not isinstance(value["asset"], str) or not value["asset"].strip():
            raise ValueError("asset must identify the underlying")
        if value["fit_status"] != "accepted" or value["source_fit_status"] not in {"converged", "synthetic_reference"}:
            raise ValueError("a converged or explicitly synthetic accepted input is required")
        for name in ("source_surface_sha256", "source_dataset_sha256", "source_curve_sha256", "artifact_sha256"):
            if not isinstance(value[name], str) or len(value[name]) != 64 or any(c not in "0123456789abcdef" for c in value[name]):
                raise ValueError(f"{name} must be a lowercase SHA256")
        for name in ("spot", "theta_floor", "epsilon", "min_denominator"):
            value[name] = _number(value[name], name, positive=True)
        value["rho"] = _number(value["rho"], "rho")
        value["eta"] = _number(value["eta"], "eta", nonnegative=True)
        if abs(value["rho"]) > 0.999 or value["epsilon"] >= 1 or value["eta"]*(1+abs(value["rho"])) > 2-value["epsilon"]:
            raise ValueError("local-vol parameters violate sufficient SSVI constraints")
        if value["min_denominator"] >= 0.25:
            raise ValueError("min_denominator must be smaller than the short-time guaranteed bound 1/4")
        nodes = value["nodes"]
        if not isinstance(nodes, list) or len(nodes) < 2:
            raise ValueError("origin and at least one positive tenor node are required")
        for i, node in enumerate(nodes):
            if set(node) != {"t", "theta", "tangent"}:
                raise ValueError("invalid theta node fields")
            for name in ("t", "theta", "tangent"):
                node[name] = _number(node[name], f"node {name}", nonnegative=True)
            if node["theta"] > 4:
                raise ValueError("ATM total variance exceeds the accepted bound 4")
            if i == 0 and (node["t"] != 0 or node["theta"] != 0):
                raise ValueError("ATM variance origin must be (0, 0)")
            if i and (node["t"] <= nodes[i-1]["t"] or node["theta"] < nodes[i-1]["theta"]):
                raise ValueError("theta nodes must have increasing time and nonnegative nondecreasing variance")
        slopes, coefficients = _pchip([node["t"] for node in nodes], [node["theta"] for node in nodes])
        if len(value["coefficients"]) != len(coefficients):
            raise ValueError("coefficient count differs from theta intervals")
        for node, slope in zip(nodes, slopes):
            if not math.isclose(node["tangent"], slope, rel_tol=1e-11, abs_tol=1e-14):
                raise ValueError("saved tangent differs from recomputed PCHIP")
        for supplied, expected in zip(value["coefficients"], coefficients):
            if set(supplied) != set(expected):
                raise ValueError("invalid polynomial coefficient fields")
            for name, wanted in expected.items():
                supplied[name] = _number(supplied[name], f"coefficient {name}")
                if not math.isclose(supplied[name], wanted, rel_tol=1e-11, abs_tol=1e-14):
                    raise ValueError("saved polynomial differs from recomputed PCHIP")
        curves = value["curves"]
        if not isinstance(curves, list) or len(curves) < 2:
            raise ValueError("dated carry curves are required")
        for i, row in enumerate(curves):
            if set(row) != {"date", "discount_factor", "forward", "source"}:
                raise ValueError("invalid carry node fields")
            row["date"] = _iso(row["date"], "carry date")
            if not isinstance(row["source"], str) or not row["source"].strip():
                raise ValueError("every carry node needs provenance")
            for name in ("discount_factor", "forward"):
                row[name] = _number(row[name], name, positive=True)
            if i == 0 and (row["date"] != value["as_of"] or row["discount_factor"] != 1 or row["forward"] != value["spot"]):
                raise ValueError("carry origin must have D(0)=1 and F(0)=spot")
            if i and row["date"] <= curves[i-1]["date"]:
                raise ValueError("carry dates must be strictly increasing")
        last_time = (date.fromisoformat(curves[-1]["date"])-date.fromisoformat(value["as_of"])).days/365
        if last_time < nodes[-1]["t"]:
            raise ValueError("carry curves do not cover the supported surface tenor")
        if not isinstance(value["coverage"], list) or not value["coverage"]:
            raise ValueError("coverage must be an explicit observation list")
        for row in value["coverage"]:
            if set(row) != {"k", "t"}:
                raise ValueError("invalid observation coverage fields")
            row["k"] = _number(row["k"], "coverage k")
            row["t"] = _number(row["t"], "coverage time", positive=True)
            if not nodes[1]["t"] <= row["t"] <= nodes[-1]["t"]:
                raise ValueError("observed coverage exceeds surface tenor support")
        if value["artifact_sha256"] != artifact_hash(value):
            raise ValueError("local-vol artifact content hash is inconsistent")
        self._artifact = value
        self._hull = _hull([(row["k"], row["t"]) for row in value["coverage"]])
        self._export_report = deepcopy(export_report)
        # The global sufficient inequalities justify the model wings. This
        # identical Python/Rust construction scan additionally rejects an
        # unstable numerical density threshold before path simulation.
        for left, right in zip(nodes, nodes[1:]):
            for t in (left["t"], (left["t"]+right["t"])/2, right["t"]):
                for k in (-1000, -100, -10, -2, -1, 0, 1, 2, 10, 100, 1000):
                    self.derivatives(k, t)

    as_of = property(lambda self: self._artifact["as_of"])
    rho = property(lambda self: self._artifact["rho"])
    eta = property(lambda self: self._artifact["eta"])
    theta_floor = property(lambda self: self._artifact["theta_floor"])
    spot = property(lambda self: self._artifact["spot"])
    currency = property(lambda self: self._artifact["currency"])
    asset = property(lambda self: self._artifact["asset"])
    nodes = property(lambda self: deepcopy(self._artifact["nodes"]))
    coverage = property(lambda self: deepcopy(self._artifact["coverage"]))
    export_report = property(lambda self: deepcopy(self._export_report))

    def _time(self, expiry):
        if isinstance(expiry, (str, date)):
            return (date.fromisoformat(_iso(expiry, "expiry"))-date.fromisoformat(self.as_of)).days/365
        return _number(expiry, "time", nonnegative=True)

    def _theta(self, t):
        t = _number(t, "time", nonnegative=True)
        nodes = self._artifact["nodes"]
        if t > nodes[-1]["t"]:
            raise ValueError("time exceeds bounded local-vol support; extrapolation is disabled")
        index = min(bisect_right([node["t"] for node in nodes], t)-1, len(nodes)-2)
        lower, upper = nodes[index], nodes[index+1]
        h = upper["t"]-lower["t"]
        u = (t-lower["t"])/h
        a, b, c, d = (self._artifact["coefficients"][index][name] for name in ("a", "b", "c", "d"))
        theta = a+u*(b+u*(c+u*d))
        delta = (upper["theta"]-lower["theta"])/h
        slope = ((1-u)**2*lower["tangent"]
                 +2*u*(1-u)*(3*delta-lower["tangent"]-upper["tangent"])
                 +u*u*upper["tangent"])
        # Exact stored endpoints avoid a polynomial roundoff sign at a knot.
        if t == lower["t"]:
            theta, slope = lower["theta"], lower["tangent"]
        elif t == upper["t"]:
            theta, slope = upper["theta"], upper["tangent"]
        if theta < 0 or slope < 0 or not math.isfinite(theta+slope):
            raise ValueError("smooth theta evaluation violated its nonnegative domain")
        return theta, slope

    def derivatives(self, k, t):
        k = _number(k, "log-forward moneyness")
        t = _number(t, "time", nonnegative=True)
        theta, slope = self._theta(t)
        w, wk, wkk, wtheta, factor, factor_k, factor_kk = _regularized(k, theta, self.rho, self.eta, self.theta_floor)
        denominator = ((1-k*factor_k/(2*factor))**2-theta*factor_k*factor_k/(4*factor)
                       -theta*theta*factor_k*factor_k/16+theta*factor_kk/2)
        wt = wtheta*slope
        values = (w, wk, wkk, wt, denominator)
        if any(not math.isfinite(item) for item in values):
            raise ValueError("local-vol derivative evaluation is nonfinite")
        if denominator <= self._artifact["min_denominator"]:
            raise ValueError("Dupire density denominator is unstable")
        raw = wt/denominator
        if not math.isfinite(raw):
            raise ValueError("local-vol variance evaluation is nonfinite")
        if raw < 0:
            raise ValueError("negative local variance; no clipping policy is enabled")
        return {"w": w, "w_k": wk, "w_kk": wkk, "w_t": wt,
                "w_theta": wtheta, "theta": theta, "theta_t": slope,
                "g": denominator, "local_variance": raw,
                "roundoff_adjusted": False, "t": t, "k": k,
                "outside_observed_coverage": not _inside((k, t), self._hull),
                "extension": "global_shifted_phi; analytic_ssvi_all_real_k"}

    def total_variance(self, k, t):
        return self.derivatives(k, t)["w"]

    def local_variance(self, k, t):
        return self.derivatives(k, t)["local_variance"]

    def _carry(self, t):
        t = self._time(t)
        self._theta(t)
        curves = self._artifact["curves"]
        times = [(date.fromisoformat(row["date"])-date.fromisoformat(self.as_of)).days/365 for row in curves]
        if t > times[-1]:
            raise ValueError("query lacks dated carry coverage")
        i = min(bisect_right(times, t)-1, len(times)-2)
        if t == 0:
            return 1.0, self.spot
        a = (t-times[i])/(times[i+1]-times[i])
        return tuple(math.exp((1-a)*math.log(curves[i][name])+a*math.log(curves[i+1][name]))
                     for name in ("discount_factor", "forward"))

    def implied_vol(self, strike, expiry):
        t = self._time(expiry)
        if t == 0:
            raise ValueError("implied volatility is undefined at zero tenor")
        strike = _number(strike, "strike", positive=True)
        _, forward = self._carry(t)
        return math.sqrt(self.total_variance(math.log(strike)-math.log(forward), t)/t)

    def black_price(self, option_type, strike, expiry):
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be call or put")
        t = self._time(expiry)
        strike = _number(strike, "strike", positive=True)
        discount, forward = self._carry(t)
        if t == 0:
            return discount*max(0.0, forward-strike if option_type == "call" else strike-forward)
        from .analytics import quantlib
        ql = quantlib()
        w = self.total_variance(math.log(strike)-math.log(forward), t)
        return float(ql.blackFormula(ql.Option.Call if option_type == "call" else ql.Option.Put,
                                     strike, forward, math.sqrt(w), discount))

    def to_dict(self):
        return deepcopy(self._artifact)

    @classmethod
    def from_dict(cls, value):
        return cls(value)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        from .snapshot import write_json
        path = Path(path)
        write_json(path, self.to_dict())
        if self._export_report is not None:
            write_json(path.with_suffix(".export_report.json"), self.export_report)
        return path

    def diagnostics(self):
        nodes = self._artifact["nodes"]
        times = sorted({node["t"] for node in nodes} | {(a["t"]+b["t"])/2 for a, b in zip(nodes, nodes[1:])})
        samples = [self.derivatives(k, t) for t in times for k in (-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0)]
        return {"artifact_sha256": self._artifact["artifact_sha256"],
                "source_fit_status": self._artifact["source_fit_status"], "fit_status": "accepted",
                "minimum_density_denominator": min(row["g"] for row in samples),
                "minimum_local_variance": min(row["local_variance"] for row in samples),
                "maximum_local_variance": max(row["local_variance"] for row in samples),
                "roundoff_adjustments": sum(row["roundoff_adjusted"] for row in samples),
                "supported_time_range": [0.0, nodes[-1]["t"]], "supported_moneyness": "all finite real k",
                "constraint_margins": {"eta": 2-self._artifact["epsilon"]-self.eta*(1+abs(self.rho)),
                                       "rho": 1-abs(self.rho)},
                "analytic_certificate": "monotone C1 theta, theta*phi increasing, 0<=dlog(theta*phi)/dlog(theta)<=1, theta*phi<=eta and theta*phi^2<=eta^2; inherited restricted SSVI sufficient constraints",
                "numerical_scan": {"times": times, "k": [-20, -5, -1, 0, 1, 5, 20],
                                   "meaning": "supplements the global analytic extension; not quote coverage"},
                "export_report": self.export_report}


def export_local_vol(surface, *, context, theta_floor=None,
                     min_denominator=DEFAULT_MIN_DENOMINATOR, allow_synthetic=False):
    """Create a separate accepted smooth artifact from a fitted SSVI surface.

    Market inputs require recorded ``fit_status='converged'``. Explicit
    ``allow_synthetic=True`` admits a deliberately synthetic test
    surface, never an incomplete market fit. The context supplies the origin,
    explicit currency/asset and bounded dated D/F curves.
    """
    if not isinstance(surface, VolSurface):
        raise ValueError("export requires a VolSurface")
    if not isinstance(allow_synthetic, bool):
        raise ValueError("allow_synthetic must be Boolean")
    metadata = surface.metadata
    parent_status = metadata.get("fit_status")
    if parent_status == "converged":
        source_status = "converged"
    elif allow_synthetic and parent_status is None:
        source_status = "synthetic_reference"
    else:
        raise ValueError("only converged fits or explicitly synthetic references may be exported")
    if context.cutoff.as_of != surface.as_of:
        raise ValueError("surface and context as_of dates differ")
    source = metadata.get("source", {})
    if source.get("underlying_ric") is not None and source["underlying_ric"] != context.asset:
        raise ValueError("surface and context underlying assets differ")
    if source.get("currency") is not None and source["currency"] != context.currency:
        raise ValueError("surface and context currencies differ")
    forward_curve = context.forward_curve
    if forward_curve is None:
        raise ValueError("local-vol export requires a forward curve")
    for row in surface.curves:
        d, f = context.discount_curve.value(row["expiry"]), forward_curve.value(row["expiry"])
        if not math.isclose(d, row["discount_factor"], rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(f, row["forward"], rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError("surface and context carry nodes are inconsistent")
    floor = surface.nodes[0]["theta"]*0.01 if theta_floor is None else _number(theta_floor, "theta_floor", positive=True)
    times = [0.0]+[node["t"] for node in surface.nodes]
    values = [0.0]+[node["theta"] for node in surface.nodes]
    slopes, coefficients = _pchip(times, values)
    carry_end = min(context.discount_curve.dates[-1], forward_curve.dates[-1])
    dates = sorted({item for item in set(context.discount_curve.dates) | set(forward_curve.dates)
                    if item <= carry_end})
    curves = [{"date": item, "discount_factor": context.discount_curve.value(item),
               "forward": forward_curve.value(item),
               "source": context.discount_curve.source+" | "+forward_curve.source} for item in dates]
    parent = surface.to_dict()
    dataset = metadata.get("dataset_sha256")
    if source_status == "synthetic_reference":
        dataset = _hash({"synthetic_reference": parent["surface_sha256"]})
    if not isinstance(dataset, str) or len(dataset) != 64 or any(c not in "0123456789abcdef" for c in dataset):
        raise ValueError("converged market input requires a valid dataset_sha256 provenance hash")
    artifact = {"schema_version": 1, "model": "regularized_ssvi_local_vol",
                "as_of": surface.as_of, "currency": context.currency, "asset": context.asset,
                "spot": context.spot, "rho": surface.rho, "eta": surface.eta,
                "theta_floor": floor, "epsilon": surface.epsilon,
                "min_denominator": min_denominator,
                "nodes": [{"t": t, "theta": theta, "tangent": tangent} for t, theta, tangent in zip(times, values, slopes)],
                "coefficients": coefficients, "curves": curves, "coverage": surface.coverage or [],
                "source_surface_sha256": parent["surface_sha256"], "source_dataset_sha256": dataset,
                "source_curve_sha256": parent["curve_sha256"], **POLICIES,
                "fit_status": "accepted", "source_fit_status": source_status}
    artifact["artifact_sha256"] = artifact_hash(artifact)
    smooth = SmoothLocalVolSurface(artifact)
    probes = sorted({(point["k"], point["t"]) for point in surface.coverage or []}
                    | {(k, node["t"]) for node in surface.nodes for k in (-0.2, 0.0, 0.2)}
                    | {(0.0, (a["t"]+b["t"])/2) for a, b in zip(surface.nodes, surface.nodes[1:])})
    differences = []
    for k, t in probes:
        d, f = smooth._carry(t)
        strike = f*math.exp(k)
        old_w = surface.total_variance(k, t, allow_wing=True)
        new_w = smooth.total_variance(k, t)
        from .analytics import quantlib
        ql = quantlib()
        old_price = float(ql.blackFormula(ql.Option.Call, strike, f, math.sqrt(old_w), d))
        new_price = smooth.black_price("call", strike, t)
        differences.append({"k": k, "t": t, "strike": strike, "original_total_variance": old_w,
                            "smooth_total_variance": new_w, "iv_change_bp": (math.sqrt(new_w/t)-math.sqrt(old_w/t))*10000,
                            "original_call_price": old_price, "smooth_call_price": new_price,
                            "price_change": new_price-old_price})
    smooth._export_report = {"schema_version": 1, "report_type": "local_vol_smooth_export",
                             "status": "accepted", "source_fit_status": source_status,
                             "source_surface_sha256": parent["surface_sha256"],
                             "artifact_sha256": artifact["artifact_sha256"],
                             "theta_floor": floor, "theta_floor_basis": "explicit" if theta_floor is not None else "0.01 * first fitted ATM variance",
                             "smoothness": "C1 theta; continuous w_t at time knots; one-sided final support boundary",
                             "quote_surface_unchanged": True, "differences": differences,
                             "max_abs_iv_change_bp": max(abs(row["iv_change_bp"]) for row in differences),
                             "max_abs_call_price_change": max(abs(row["price_change"]) for row in differences),
                             "meaning": "smooth-model changes relative to the original fit; these differences are separate from original market calibration residuals"}
    return smooth


def finite_difference_price(surface, option_type, strike, expiry, *,
                            space_points=401, time_steps=400, x_bounds=(-2.0, 2.0)):
    """Independent backward PDE in x=log(S/F), with Rannacher start-up.

    Solves u_t + local_variance(x,t)/2 * (u_xx-u_x)=0 without discount;
    multiplication by the one supplied terminal D(T)F(T) occurs once. The
    declared spatial boundary truncation is a numerical approximation and is
    reported explicitly. No constant-volatility fallback or clipping is used.
    """
    import numpy as np
    from scipy.linalg import solve_banded
    if not isinstance(surface, SmoothLocalVolSurface):
        raise ValueError("PDE requires a validated SmoothLocalVolSurface")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 3 for n in (space_points, time_steps)):
        raise ValueError("space_points/time_steps must be integers >=3")
    strike = _number(strike, "strike", positive=True)
    maturity = surface._time(expiry)
    surface._theta(maturity)
    if maturity <= 0:
        raise ValueError("PDE requires positive maturity")
    xmin, xmax = [_number(item, "x boundary") for item in x_bounds]
    if not xmin < 0 < xmax:
        raise ValueError("PDE spatial domain must straddle x=0")
    started = time.perf_counter()
    discount, forward = surface._carry(maturity)
    moneyness = strike/forward
    x = np.linspace(xmin, xmax, space_points)
    dx = float(x[1]-x[0])
    if dx >= 2:
        raise ValueError("PDE dx must be below 2 for nonnegative off-diagonal coefficients")
    terminal = np.exp(x)-moneyness if option_type == "call" else moneyness-np.exp(x)
    u = np.maximum(terminal, 0)
    low, high = float(u[0]), float(u[-1])
    # Keep the local model's C1 knots on the time grid, while preserving all
    # requested uniform mesh points. First two reverse steps use fully
    # implicit half-steps to damp the payoff kink (Rannacher smoothing).
    times = sorted(set(np.linspace(0, maturity, time_steps+1)) | {
        node["t"] for node in surface.nodes if 0 < node["t"] < maturity})
    reverse = list(zip(times[-2::-1], times[:0:-1]))
    steps = 0
    for j, (left, right) in enumerate(reverse):
        substeps = [(left+(right-left)/2, right), (left, left+(right-left)/2)] if j < 2 else [(left, right)]
        alpha = 1.0 if j < 2 else 0.5
        for a, b in substeps:
            dt = b-a
            coeff = np.array([surface.local_variance(float(k), (a+b)/2) for k in x[1:-1]])
            lower = coeff/2*(1/dx**2+1/(2*dx))
            upper = coeff/2*(1/dx**2-1/(2*dx))
            diagonal = -coeff/dx**2
            rhs = (1+(1-alpha)*dt*diagonal)*u[1:-1]
            rhs += (1-alpha)*dt*(lower*u[:-2]+upper*u[2:])
            rhs[0] += alpha*dt*lower[0]*low
            rhs[-1] += alpha*dt*upper[-1]*high
            band = np.zeros((3, space_points-2))
            band[0, 1:] = -alpha*dt*upper[:-1]
            band[1] = 1-alpha*dt*diagonal
            band[2, :-1] = -alpha*dt*lower[1:]
            u[1:-1] = solve_banded((1, 1), band, rhs, check_finite=True)
            u[0], u[-1] = low, high
            steps += 1
    price = float(np.interp(0.0, x, u))*discount*forward
    if not math.isfinite(price) or price < 0:
        raise ValueError("PDE reference produced a nonfinite or negative price")
    reference = surface.black_price(option_type, strike, maturity)
    return {"price": price, "surface_black_price": reference, "price_error": price-reference,
            "option_type": option_type, "strike": strike, "t": maturity,
            "space_points": space_points, "requested_time_steps": time_steps, "actual_time_steps": steps,
            "x_bounds": [xmin, xmax], "dx": dx, "scheme": "backward Crank-Nicolson with two Rannacher steps",
            "boundary_policy": "fixed normalized intrinsic asymptotes; explicit truncated spatial domain",
            "generator": "0.5 * local_variance * (dxx - dx)", "discounting": "terminal D(T)F(T) once",
            "elapsed_seconds": time.perf_counter()-started,
            "artifact_sha256": surface.to_dict()["artifact_sha256"], "declared_tolerances": deepcopy(REFERENCE_TOLERANCES)}
