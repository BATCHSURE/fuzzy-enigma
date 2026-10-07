"""Credential-free, versioned JSON snapshots and explicit historical curves."""

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import tempfile

from . import SCHEMA_VERSION

_SECRET_KEYS = {"app_key", "appkey", "api_key", "apikey", "authorization",
                "password", "client_secret", "access_token", "refresh_token"}


def redact(value):
    """Strip authentication fields even from nested provider diagnostics."""
    if isinstance(value, dict):
        return {str(k): ("[REDACTED]" if str(k).lower().replace("-", "_") in _SECRET_KEYS
                         else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        # App keys can appear inside SDK error strings as well as named fields.
        value = re.sub(r"(?i)(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", "[REDACTED]", value)
        return re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(redact(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".snapshot-", delete=False) as f:
        temp = Path(f.name)
        f.write(data)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_snapshot(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported snapshot schema_version; expected 1")
    if not isinstance(value.get("source"), dict):
        raise ValueError("Snapshot must include source metadata")
    if not isinstance(value.get("quotes"), list):
        raise ValueError("Snapshot quotes must be a list")
    return redact(value)


def dataset_hash(snapshot):
    # Exclude the hash field itself; all financial inputs remain part of the hash.
    value = redact(snapshot)
    value = {k: v for k, v in value.items() if k not in {"dataset_sha256", "versions"}}
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def copy_evidence(snapshot, original_path, output):
    """Carry a verified raw-response sidecar into a replay's artifact directory."""
    source = snapshot["source"]
    filename = source.get("raw_response_file")
    if not filename:
        return
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("raw_response_file must be a relative path inside the snapshot directory")
    original_dir = Path(original_path).resolve().parent
    raw_path = (original_dir / relative).resolve()
    if not raw_path.is_relative_to(original_dir):
        raise ValueError("Raw response sidecar resolves outside the snapshot directory")
    if not raw_path.is_file():
        source["raw_response_status"] = "missing_sidecar"
        source["original_raw_response_file"] = filename
        source.pop("raw_response_file", None)
        return
    expected = source.get("raw_response_sha256")
    actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError("Raw response sidecar hash does not match the snapshot")
    destination = Path(output) / "raw_responses.json"
    write_json(destination, json.loads(raw_path.read_text(encoding="utf-8")))
    source["raw_response_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    source["raw_response_file"] = destination.name
    source["raw_response_status"] = "verified"


def versions():
    packages = ("fuzzy-enigma", "lseg-data", "QuantLib", "scipy", "numpy", "pandas", "matplotlib")
    result = {}
    for name in packages:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def read_curves(path, as_of):
    """CSV: as_of, expiry, discount_factor, forward, source (no inferred zeros)."""
    from datetime import date

    required = {"as_of", "expiry", "discount_factor", "forward", "source"}
    result = []
    seen = set()
    date.fromisoformat(as_of)
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Curve CSV requires: " + ", ".join(sorted(required)))
        for row in reader:
            expiry = row["expiry"].strip()
            date.fromisoformat(expiry)
            if row["as_of"].strip() != as_of:
                raise ValueError("Every curve row must match the snapshot as_of date")
            if expiry in seen:
                raise ValueError(f"Duplicate curve expiry: {expiry}")
            discount, forward = float(row["discount_factor"]), float(row["forward"])
            if not all(math.isfinite(x) and x > 0 for x in (discount, forward)):
                raise ValueError("Curve discount_factor and forward must be finite and positive")
            source = row["source"].strip()
            if not source:
                raise ValueError("Every curve row needs a source")
            result.append(dict(as_of=as_of, expiry=expiry, discount_factor=discount,
                               forward=forward, source=source))
            seen.add(expiry)
    if not result:
        raise ValueError("Curve CSV is empty")
    return result
