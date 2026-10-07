"""Freeze a small independent reference fixture from a validated snapshot.

This is a manual maintenance command. CI reads expected values, never rebuilds
them and never connects to LSEG. Market Close residuals are diagnostics only.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json

from .analytics import black_price, quantlib, validate_snapshot
from .calibration import heston_price
from .snapshot import dataset_hash, read_snapshot, write_json


def make_fixture(snapshot, parameters=None):
    validated = validate_snapshot(snapshot)
    rows = validated["accepted"]
    if not rows:
        raise ValueError("Cannot generate reference cases without valid quotes and same-date curves")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["expiry"]][row["strike"]].append(row)
    expiries = sorted(grouped)
    expiries = [expiries[0], expiries[-1]] if len(expiries) > 1 else expiries
    cases = []
    for expiry in expiries:
        strikes = sorted(grouped[expiry])
        selected = sorted({strikes[0], strikes[len(strikes) // 2], strikes[-1]})
        for strike in selected:
            for kind in ("call", "put"):
                candidates = [row for row in grouped[expiry][strike] if row["kind"] == kind]
                if not candidates:
                    continue
                row = min(candidates, key=lambda item: item["ric"])
                fields = ("ric", "as_of", "expiry", "expiry_utc", "kind", "strike", "spot", "t",
                          "r", "q", "discount_factor", "forward", "curve_source", "close", "close_date", "iv",
                          "exercise_style", "settlement", "settlement_type", "currency", "premium_unit",
                          "contract_multiplier", "terms_source", "expiry_source_value", "expiry_timestamp_status")
                case = {name: row[name] for name in fields if name in row}
                case.update(sigma=0.20, expected_black=black_price(row, 0.20),
                            analytic_tolerance=max(1e-10, row["spot"] * 1e-11),
                            mc={"paths": 60_000, "steps": 1, "seed": 42, "standard_errors": 5})
                if parameters:
                    case["heston_parameters"] = parameters
                    case["expected_heston"] = heston_price(row, parameters)
                cases.append(case)
    return {
        "fixture_schema_version": 1,
        "source": {key: value for key, value in snapshot["source"].items() if key != "raw_response_file"},
        "dataset_sha256": dataset_hash(snapshot),
        "reference": {"library": "QuantLib", "version": quantlib().__version__,
                      "black_engine": "BlackCalculator", "heston_engine": "AnalyticHestonEngine",
                      "heston_integration_order": 144, "generated_at_utc": datetime.now(timezone.utc).isoformat()},
        "conventions": {"day_count": "ACT/365F", "units": "index_points", "volatility": "annual_decimal",
                        "meaning": "Frozen independent mathematical references; market Close is diagnostic only"},
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heston-parameters", help="Calibration JSON containing parameters")
    args = parser.parse_args(argv)
    snapshot = read_snapshot(args.snapshot)
    parameters = None
    if args.heston_parameters:
        with open(args.heston_parameters, encoding="utf-8") as f:
            result = json.load(f)
        parameters = result.get("parameters", result.get("params"))
        if not parameters:
            raise ValueError("Calibration file does not contain fitted parameters")
    fixture = make_fixture(snapshot, parameters)
    write_json(args.output, fixture)
    print(f"Saved {len(fixture['cases'])} frozen reference cases to {args.output}")


if __name__ == "__main__":
    main()
