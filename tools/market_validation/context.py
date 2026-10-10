"""Build native dated valuation contexts from verified offline snapshots.

The captured positive-tenor curve values are preserved. Only the as-of
discount/forward anchors are derived. This bridge does not fetch data, fit a
surface, choose model dynamics, or persist a new artifact format.
"""

from collections.abc import Mapping
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re

from .snapshot import dataset_hash, read_snapshot, redact


def _iso_date(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("{} must be an ISO YYYY-MM-DD date".format(name))
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("{} is not a valid calendar date".format(name)) from exc


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError("{} must be positive and finite".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be positive and finite".format(name)) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("{} must be positive and finite".format(name))
    return result


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def context_from_snapshot(snapshot_or_path, *, currency, time_grid=None,
                          phase="after_fixing"):
    """Return ``(ValuationContext, audit)`` from an existing market snapshot.

    ``currency`` is an explicit caller declaration, checked against any
    currencies recorded in the snapshot. ``time_grid`` is optional and, when
    supplied, is checked against the snapshot date and curve coverage. The
    supplied grid is attached to the returned context as its pricing default.

    A valid ``dataset_sha256`` is required and verified before native objects
    are constructed. Audit hashes identify the parent snapshot and the derived
    anchored curves. They do not certify vendor units or intraday cutoffs.
    """
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("currency must be an explicit three-letter uppercase code")
    if phase not in {"before_fixing", "after_fixing"}:
        raise ValueError("phase must be before_fixing or after_fixing")
    if isinstance(snapshot_or_path, Mapping):
        snapshot = redact(deepcopy(dict(snapshot_or_path)))
    elif isinstance(snapshot_or_path, (str, Path)):
        snapshot = read_snapshot(snapshot_or_path)
    else:
        raise ValueError("snapshot must be a mapping or a snapshot path")
    if snapshot.get("schema_version") != 1 or not isinstance(snapshot.get("source"), dict):
        raise ValueError("snapshot must have schema_version 1 and source metadata")
    if not isinstance(snapshot.get("quotes"), list):
        raise ValueError("snapshot quotes must be a list")
    expected = snapshot.get("dataset_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("snapshot must include a valid dataset_sha256")
    try:
        actual = dataset_hash(snapshot)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("snapshot contains nonfinite or non-JSON financial inputs") from exc
    if expected != actual:
        raise ValueError("snapshot dataset_sha256 does not match its contents")

    source = snapshot["source"]
    if phase == "before_fixing" and source.get("price_basis") == "close":
        raise ValueError("an end-of-day Close snapshot cannot supply a before_fixing context")
    as_of = _iso_date(source.get("as_of"), "source.as_of")
    asset = source.get("underlying_ric")
    if not isinstance(asset, str) or not asset.strip():
        raise ValueError("source.underlying_ric must identify the underlying asset")
    spot = _positive(snapshot.get("spot"), "spot")
    declared = [snapshot.get("currency"), source.get("currency")]
    declared.extend(row.get("currency") for row in snapshot["quotes"] if isinstance(row, dict))
    if any(value is not None and value != currency for value in declared):
        raise ValueError("caller currency conflicts with snapshot currency metadata")
    rows = snapshot.get("curves")
    if not isinstance(rows, list) or not rows:
        raise ValueError("snapshot must include dated discount and forward curves")
    nodes = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("curve nodes must be mappings")
        if row.get("as_of") != as_of.isoformat():
            raise ValueError("curve nodes must share the snapshot as_of date")
        if row.get("currency") is not None and row["currency"] != currency:
            raise ValueError("caller currency conflicts with curve currency metadata")
        expiry = _iso_date(row.get("expiry"), "curve expiry")
        if expiry <= as_of or expiry in seen:
            raise ValueError("curve expiries must be unique dates after as_of")
        seen.add(expiry)
        node_source = row.get("source")
        if not isinstance(node_source, str) or not node_source.strip():
            raise ValueError("each captured curve node must retain its source")
        nodes.append({"date": expiry.isoformat(),
                      "discount_factor": _positive(row.get("discount_factor"), "discount_factor"),
                      "forward": _positive(row.get("forward"), "forward"),
                      "source": node_source})
    nodes.sort(key=lambda row: row["date"])
    anchor = {"date": as_of.isoformat(), "discount_factor": 1.0,
              "forward": spot, "source": "derived as-of anchor"}
    anchored = [anchor] + nodes
    curve_inputs = {"as_of": as_of.isoformat(), "currency": currency, "asset": asset,
                    "spot": spot, "nodes": anchored, "day_count": "ACT/365F",
                    "interpolation": "log_linear", "extrapolation": "reject"}
    curve_hash = _hash(curve_inputs)
    audit = {"parent_dataset_sha256": actual,
             "parent_capture_dataset_sha256": source.get("capture_dataset_sha256"),
             "derived_curve_sha256": curve_hash,
             "as_of": as_of.isoformat(), "currency": currency,
             "currency_basis": "caller_declared_checked_against_recorded_metadata",
             "asset": asset, "spot": spot, "cutoff_phase": phase,
             "source": deepcopy(source), "day_count": "ACT/365F",
             "interpolation": "log_linear", "extrapolation": "reject",
             "origin_anchor": deepcopy(anchor), "captured_nodes": deepcopy(nodes),
             "coverage_end": nodes[-1]["date"]}

    # Importing this helper never opens a supplier session; only constructing
    # the context requires the project's native extension.
    import fuzzy_enigma as fe
    dates = [row["date"] for row in anchored]
    discounts = [row["discount_factor"] for row in anchored]
    forwards = [row["forward"] for row in anchored]
    curve_source = "offline snapshot {} / derived anchored curves {}".format(actual, curve_hash)
    discount = fe.DiscountCurve(as_of.isoformat(), currency, dates, discounts, curve_source)
    forward = fe.ForwardCurve(as_of.isoformat(), currency, asset, dates, forwards, curve_source)
    cutoff = fe.ValuationCutoff(as_of.isoformat(), phase)
    context = fe.ValuationContext(cutoff, spot, currency, asset, discount, forward, curve_source)
    if time_grid is not None:
        grid_as_of = str(time_grid.as_of)
        if grid_as_of != as_of.isoformat():
            raise ValueError("time_grid as_of must match the snapshot date")
        grid_dates = [str(value) for value in time_grid.fixing_dates]
        grid_times = list(time_grid.times)
        last_time = (_iso_date(nodes[-1]["date"], "coverage end") - as_of).days / 365.0
        if not grid_times or grid_times[0] != 0.0:
            raise ValueError("time_grid must start at the snapshot as_of date")
        if grid_times[-1] > last_time + 1e-12:
            raise ValueError("time_grid exceeds captured curve coverage")
        grid_inputs = {"as_of": grid_as_of, "fixing_dates": grid_dates,
                       "times": grid_times, "day_count": "ACT/365F"}
        audit["time_grid_sha256"] = _hash(grid_inputs)
        audit["time_grid"] = grid_inputs
        context = context.with_grid(time_grid)
    audit["context_sha256"] = _hash({"derived_curve_sha256": curve_hash,
                                       "cutoff_phase": phase,
                                       "time_grid_sha256": audit.get("time_grid_sha256")})
    return context, audit
