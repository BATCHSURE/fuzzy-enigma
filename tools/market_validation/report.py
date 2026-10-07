"""Static HTML, CSV/JSON and masked, non-extrapolated volatility charts."""

from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
import tempfile

from .analytics import implied_volatility
from .calibration import heston_price, representatives


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _vendor_points(validated):
    """Overlay only explicitly compatible carry, date and vol conventions."""
    surface = validated.get("vendor_surface")
    if not surface:
        return []
    curves = {curve["expiry"]: curve for curve in validated.get("curves", [])}
    result = []
    for point in surface.get("points", []):
        curve = curves.get(point["expiry"])
        if curve is None:
            continue
        confirmed = surface.get("curve_compatible") is True
        if "forward" in point and "discount_factor" in point:
            try:
                confirmed = math.isclose(float(point["forward"]), curve["forward"], rel_tol=1e-8) and math.isclose(float(point["discount_factor"]), curve["discount_factor"], rel_tol=1e-8)
            except (TypeError, ValueError, OverflowError):
                confirmed = False
        if confirmed:
            settings = validated.get("settings", {})
            moneyness = point["strike"] / curve["forward"]
            if settings.get("min_moneyness", 0.8) <= moneyness <= settings.get("max_moneyness", 1.2):
                result.append({**point, "forward": curve["forward"], "discount_factor": curve["discount_factor"]})
    return result


def _plots(validated, calibration, output):
    rows = representatives(validated["accepted"])
    if not rows:
        return [], ["No accepted OTM observations available for plotting."]
    plot_cache = Path(tempfile.mkdtemp(prefix="fuzzy-market-mpl-"))
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache / "cache"))
    try:
        import numpy as np
        from scipy.interpolate import LinearNDInterpolator
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Reports require optional numpy, scipy and matplotlib dependencies.") from exc
    predictions = representatives(calibration.get("predictions", [])) if calibration else []
    predicted = {row["group"]: row for row in predictions}
    excluded_expiries = {item["expiry"] for item in calibration.get("excluded_expiries", [])} if calibration else set()
    rows = [{**row, "split": "excluded_expiry" if row["expiry"] in excluded_expiries else row["split"]}
            for row in rows]
    vendor = _vendor_points(validated)
    warnings, artifacts = [], []
    vendor_metadata = validated.get("vendor_surface") or {}
    if vendor_metadata and not vendor_metadata.get("units_verified_by_response", True):
        warnings.append("LSEG overlay uses the recorded SDK example convention (percent, European Black IV); the response did not declare units. It is a fitted-model diagnostic, not an independent pricing reference.")
    if validated.get("vendor_surface") and not vendor:
        warnings.append("Vendor surface was retained, but no carry-compatible points were available for overlay.")
    expiries = sorted({row["expiry"] for row in rows})
    shown_expiries = expiries
    if len(expiries) > 6:
        shown_expiries = [expiries[index] for index in np.unique(np.linspace(0, len(expiries) - 1, 6).astype(int))]
    fig, axes = plt.subplots(len(shown_expiries), 1, figsize=(10, 3.2 * len(shown_expiries)), squeeze=False)
    for ax, expiry in zip(axes[:, 0], shown_expiries):
        selected = [row for row in rows if row["expiry"] == expiry]
        for split, marker in (("train", "o"), ("holdout", "s"), ("excluded_expiry", "D")):
            group = [row for row in selected if row["split"] == split]
            if not group:
                continue
            ax.scatter([row["strike"] / row["forward"] for row in group], [100 * row["iv"] for row in group],
                       marker=marker, label="Close IV " + split)
        fitted = [predicted[row["group"]] for row in selected if row["group"] in predicted and predicted[row["group"]].get("model_iv") is not None]
        if fitted:
            ax.plot([row["strike"] / row["forward"] for row in fitted], [100 * row["model_iv"] for row in fitted], "-", label="Calibrated Heston")
        points = sorted([point for point in vendor if point["expiry"] == expiry], key=lambda point: point["strike"])
        if points:
            ax.scatter([point["strike"] / point["forward"] for point in points], [100 * point["iv"] for point in points], marker="x", label="Compatible vendor Black IV")
        ax.set(title=expiry, xlabel="Strike / forward", ylabel="Black IV (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "smiles.png", dpi=160)
    plt.close(fig)
    artifacts.append("smiles.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    nearest = [min((row for row in rows if row["expiry"] == expiry), key=lambda row: abs(math.log(row["strike"] / row["forward"]))) for expiry in expiries]
    ax.plot([row["t"] for row in nearest], [100 * row["iv"] for row in nearest], "o-", label="Nearest observed ATM Close IV")
    params = calibration.get("parameters", calibration.get("params")) if calibration else None
    if params:
        term = []
        for row in nearest:
            atm = {**row, "strike": row["forward"], "kind": "call"}
            try:
                term.append((row["t"], implied_volatility(atm, heston_price(atm, params))))
            except (ValueError, RuntimeError):
                warnings.append("A calibrated ATM IV was unavailable; its term-structure point was omitted.")
        if term:
            ax.plot([point[0] for point in term], [100 * point[1] for point in term], "s-", label="Calibrated exact ATM Heston IV")
    ax.set(xlabel="ACT/365F maturity (years)", ylabel="Black IV (%)", title="ATM term structure; no maturity extrapolation")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "term_structure.png", dpi=160)
    plt.close(fig)
    artifacts.append("term_structure.png")

    # Linear interpolation of total variance, not volatility, restricted
    # to the observed (log forward moneyness, maturity) convex hull.
    points = np.array([(math.log(row["strike"] / row["forward"]), row["t"]) for row in rows])
    if len(rows) >= 3 and np.linalg.matrix_rank(points - points[0]) == 2:
        surfaces = [("Market Close IV", np.array([row["iv"] ** 2 * row["t"] for row in rows]))]
        if all(row["group"] in predicted and predicted[row["group"]].get("model_iv") is not None for row in rows):
            surfaces.append(("Heston IV at observed knots", np.array([predicted[row["group"]]["model_iv"] ** 2 * row["t"] for row in rows])))
        log_grid, time_grid = np.meshgrid(np.linspace(points[:, 0].min(), points[:, 0].max(), 60), np.linspace(points[:, 1].min(), points[:, 1].max(), 45))
        fig = plt.figure(figsize=(8 * len(surfaces), 5))
        displayed_vols = [row["iv"] for row in rows] + [row["model_iv"] for row in predictions if row.get("model_iv") is not None]
        z_ceiling = max(1.0, 100 * max(displayed_vols)) * 1.05
        for index, (title, total_variance) in enumerate(surfaces, start=1):
            interpolated = LinearNDInterpolator(points, total_variance, fill_value=np.nan)(log_grid, time_grid)
            iv = np.sqrt(np.maximum(interpolated, 0) / time_grid)
            ax = fig.add_subplot(1, len(surfaces), index, projection="3d")
            ax.plot_surface(np.exp(log_grid), time_grid, 100 * iv, cmap="viridis", vmin=0, vmax=z_ceiling, linewidth=0, alpha=0.9)
            ax.set(xlabel="Strike / forward", ylabel="Maturity (years)", zlabel="Black IV (%)", title=title)
            ax.set_zlim(0, z_ceiling)
        fig.suptitle("Linear total-variance interpolation inside observed hull; not an arbitrage-free projection")
        fig.tight_layout()
        fig.savefig(output / "surface.png", dpi=160)
        plt.close(fig)
        artifacts.append("surface.png")
    else:
        warnings.append("A 2D surface requires observations spanning at least two maturities and moneyness coordinates.")

    if predictions:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for split, marker in (("train", "o"), ("holdout", "s"), ("excluded_expiry", "D")):
            selected = [row for row in predictions if row["split"] == split]
            if not selected:
                continue
            axes[0].scatter([row["strike"] / row["forward"] for row in selected], [row["price_error"] for row in selected], marker=marker, label=split)
            iv_rows = [row for row in selected if row.get("iv_error_bp") is not None]
            axes[1].scatter([row["strike"] / row["forward"] for row in iv_rows], [row["iv_error_bp"] for row in iv_rows], marker=marker, label=split)
        for ax in axes:
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Strike / forward")
            ax.grid(alpha=0.2)
            ax.legend()
        axes[0].set_ylabel("Model minus Close (index points)")
        axes[1].set_ylabel("Model minus Close IV (basis points)")
        fig.tight_layout()
        fig.savefig(output / "residuals.png", dpi=160)
        plt.close(fig)
        artifacts.append("residuals.png")
    return artifacts, warnings


def write_report(validated, calibration_or_none, numerical_or_none, output):
    """Write an auditable report directory and return its HTML path."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "validation.json", validated)
    _write_json(output / "calibration.json", calibration_or_none)
    _write_json(output / "numerical_validation.json", numerical_or_none)
    residual_rows = calibration_or_none.get("predictions", []) if calibration_or_none else validated["accepted"]
    _write_csv(output / "residuals.csv", residual_rows,
               ["ric", "expiry", "strike", "kind", "reference_kind", "close", "iv", "model_price", "price_error",
                "model_iv", "iv_error_bp", "iv_error_reason", "uncertainty_price", "uncertainty_source", "weighted_error",
                "split", "calibration_role", "group", "bid_ask_usable", "inside_historical_bid_ask", "forward", "discount_factor", "t", "r", "q"])
    # The CSV includes independent model IV differences, not a Close-IV
    # round trip presented as a market-validation result.
    numerical_fields = ["ric", "expiry", "strike", "kind", "model", "paths", "samples", "steps", "seed", "reference_price",
                        "mc_price", "std_error", "price_error", "z_score", "core_analytic_price", "core_analytic_error",
                        "within_sampling_tolerance", "reference_iv", "mc_iv", "iv_error_bp", "iv_error_reason"]
    _write_csv(output / "numerical_validation.csv", numerical_or_none.get("records", []) if numerical_or_none else [], numerical_fields)
    images, warnings = _plots(validated, calibration_or_none, output)
    summary = {"source": validated["source"], "spot": validated["spot"], "accepted_quotes": len(validated["accepted"]),
               "dataset_sha256": validated.get("dataset_sha256"), "versions": validated.get("versions"),
               "timings_seconds": validated.get("timings_seconds"),
               "vendor_surface_metadata": {key: value for key, value in (validated.get("vendor_surface") or {}).items() if key != "points"},
               "calibration_settings": calibration_or_none.get("settings") if calibration_or_none else None,
               "numerical_settings": numerical_or_none.get("settings") if numerical_or_none else None,
               "rejected_quotes": len(validated["rejected"]), "settings": validated.get("settings"),
               "calibration_status": calibration_or_none.get("status") if calibration_or_none else "not_requested",
               "parameters": calibration_or_none.get("parameters") if calibration_or_none else None,
               "train": calibration_or_none.get("train") if calibration_or_none else None,
               "strike_holdout": calibration_or_none.get("holdout") if calibration_or_none else None,
               "all_train_counts": calibration_or_none.get("train_counts") if calibration_or_none else None,
               "selected_train_counts": calibration_or_none.get("selected_train_counts") if calibration_or_none else None,
               "excluded_expiries": calibration_or_none.get("excluded_expiries") if calibration_or_none else None,
               "numerical_status": numerical_or_none.get("status") if numerical_or_none else "not_requested",
               "numerical_summaries": numerical_or_none.get("summaries") if numerical_or_none else None,
               "warnings": warnings, "diagnostics": validated["diagnostics"]}
    _write_json(output / "summary.json", summary)
    links = ["summary.json", "validation.json", "calibration.json", "numerical_validation.json", "residuals.csv", "numerical_validation.csv"]
    figures = "\n".join(f'<figure><img src="{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in images)
    link_html = " · ".join(f'<a href="{html.escape(name)}">{html.escape(name)}</a>' for name in links)
    page = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPX EOD market validation</title><style>
body{{font:16px/1.5 system-ui,sans-serif;color:#17212b;max-width:1200px;margin:40px auto;padding:0 24px}}
pre{{background:#f3f5f7;padding:20px;overflow:auto}}img{{max-width:100%;height:auto}}figure{{margin:32px 0}}
a{{color:#075b94}}.note{{border-left:4px solid #d49721;padding:12px 18px;background:#fff8e6}}
</style><h1>SPX EOD market validation</h1>
<p class="note">Observed references are Close prices, not bid/ask mid prices. Missing historical bid/ask does not imply a tradable price interval.
Calibration residuals measure market fit; independent core-versus-QuantLib results measure sampling and discretization or implementation error.
The every-fifth-strike holdout tests interpolation within observed expiries. Surface plots interpolate total variance inside the observed hull, without extrapolation or an arbitrage-free claim.</p>
<p>{link_html}</p><h2>Run summary</h2><pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))}</pre>
{figures}</html>'''
    path = output / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
