"""Run from the repository root: python -m tools.market_validation --help."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

from .snapshot import copy_evidence, dataset_hash, read_curves, read_snapshot, redact, versions, write_json


def _positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return result


def _sequence(value):
    try:
        result = tuple(_positive(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("Sequence must not be empty")
    return result


def _seed_sequence(value):
    try:
        result = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integer seeds") from exc
    if not result or len(set(result)) != len(result) or any(x < 0 or x >= 2**64 for x in result):
        raise argparse.ArgumentTypeError("Seeds must be distinct unsigned 64-bit integers")
    return result


def parser():
    result = argparse.ArgumentParser(description="SPXW EOD validation, IV surfaces and Heston calibration")
    commands = result.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check Desktop, SPX, option discovery/history and IPA access")
    doctor.add_argument("--prompt", action="store_true", help="Read an app key through hidden input")
    doctor.add_argument("--output", type=Path, help="Save redacted doctor.json")
    for name in ("fetch", "validate", "calibrate", "report", "run"):
        p = commands.add_parser(name, help={
            "fetch": "Fetch a credential-free LSEG EOD snapshot",
            "validate": "Validate an offline snapshot and compare GBM MC to QuantLib",
            "calibrate": "Fit Heston with deterministic pricing and save a report",
            "report": "Rebuild the complete report from an offline snapshot",
            "run": "Fetch or replay, validate, calibrate and report",
        }[name])
        p.add_argument("--snapshot", type=Path, help="Replay a snapshot without connecting to LSEG")
        p.add_argument("--as-of", help="Requested YYYY-MM-DD (default: last completed published session)")
        p.add_argument("--curves", type=Path, help="Explicit historical curves CSV")
        p.add_argument("--output", type=Path, help="Artifact directory (default: artifacts/market_validation/<UTC timestamp>)")
        p.add_argument("--paths", type=_positive, default=100_000)
        p.add_argument("--steps", type=_sequence, default=(64, 256, 1024))
        p.add_argument("--seeds", type=_seed_sequence, default=(42, 43, 44))
        p.add_argument("--starts", type=_positive, default=8)
        p.add_argument("--max-nfev", type=_positive, default=500)
        p.add_argument("--no-calibration", action="store_true")
        p.add_argument("--no-mc", action="store_true")
        if name in {"fetch", "run"}:
            p.add_argument("--verify-vendor-surface", action="store_true",
                           help="Request typed IPA supplier-volatility evidence; requires FinancialContracts content access")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    started = perf_counter()
    timings = {}
    output = args.output or Path("artifacts/market_validation") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        if args.command == "doctor":
            from .lseg import doctor
            status = doctor(prompt=args.prompt)
            if args.output:
                write_json(output / "doctor.json", status)
            print(json.dumps(redact(status), indent=2, allow_nan=False))
            return 0 if status.get("status") in ("ok", "partial") else 1
        if args.command == "fetch" and args.snapshot:
            raise ValueError("fetch does not accept --snapshot; use run or validate for replay")
        if args.command == "report" and not args.snapshot:
            raise ValueError("report requires --snapshot and never connects to LSEG")
        if args.snapshot and getattr(args, "verify_vendor_surface", False):
            raise ValueError("--verify-vendor-surface requires live fetch; offline replay never contacts LSEG")
        if args.snapshot:
            snapshot = read_snapshot(args.snapshot)
            copy_evidence(snapshot, args.snapshot, output)
            if args.as_of and args.as_of != snapshot["source"].get("as_of"):
                raise ValueError("--as-of differs from the snapshot date")
            if args.curves:
                snapshot["curves"] = read_curves(args.curves, snapshot["source"]["as_of"])
        else:
            from .lseg import fetch_snapshot
            snapshot = fetch_snapshot(as_of=args.as_of, curves_path=args.curves, output=output,
                                      verify_vendor_surface=getattr(args, "verify_vendor_surface", False))
        snapshot["dataset_sha256"] = dataset_hash(snapshot)
        timings["load_or_fetch"] = perf_counter() - started
        snapshot["versions"] = versions()
        write_json(output / "snapshot.json", snapshot)
        if args.command == "fetch":
            print(f"Snapshot: {output / 'snapshot.json'} ({len(snapshot['quotes'])} quotes)")
            return 0
        from .analytics import validate_snapshot
        from .calibration import calibrate, numerical_validation
        from .report import write_report

        validation_started = perf_counter()
        validated = validate_snapshot(snapshot)
        timings["quote_validation"] = perf_counter() - validation_started
        validated["dataset_sha256"] = snapshot["dataset_sha256"]
        validated["versions"] = snapshot["versions"]
        write_json(output / "validated.json", validated)
        if not validated["accepted"]:
            write_report(validated, None, None, output)
            raise ValueError("No valid quotes; see validated.json for rejection reasons")
        calibration = None
        numerical = None
        if args.command in {"calibrate", "report", "run"} and not args.no_calibration:
            calibration_started = perf_counter()
            calibration = calibrate(validated, starts=args.starts, max_nfev=args.max_nfev)
            timings["calibration"] = perf_counter() - calibration_started
            write_json(output / "calibration.json", calibration)
        if args.command in {"validate", "report", "run"} and not args.no_mc:
            numerical_started = perf_counter()
            params = calibration.get("parameters") if calibration else None
            numerical = numerical_validation(validated, params=params, paths=args.paths,
                                               steps=args.steps, seeds=args.seeds)
            write_json(output / "numerical_validation.json", numerical)
            timings["numerical_validation"] = perf_counter() - numerical_started
        timings["before_report"] = perf_counter() - started
        validated["timings_seconds"] = timings
        write_json(output / "validated.json", validated)
        path = write_report(validated, calibration, numerical, output)
        print(f"Accepted {len(validated['accepted'])}; rejected {len(validated['rejected'])}")
        print(f"Report: {path}")
        if calibration:
            print("Calibration status: " + str(calibration.get("status", "unknown")))
        return 0 if not calibration or calibration.get("status") == "converged" else 1
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        error = {"status": "error", "type": type(exc).__name__, "message": str(exc)}
        if hasattr(exc, "code"):
            error["code"] = exc.code
        if hasattr(exc, "diagnostics"):
            error["diagnostics"] = exc.diagnostics
        write_json(output / "error.json", error)
        print(json.dumps(redact(error), indent=2), file=sys.stderr)
        print(f"Diagnostics: {output / 'error.json'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
