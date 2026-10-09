"""Static HTML, CSV/JSON and masked, non-extrapolated volatility charts."""

from __future__ import annotations

import csv
import html
import json
import math
import os
from pathlib import Path
import tempfile
from collections import defaultdict

from .analytics import implied_volatility
from .calibration import heston_price, representatives


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True, allow_nan=False)
                          if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


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


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _surface_label(surface_fit):
    if surface_fit.get("status") == "converged":
        return "Local SSVI fit"
    return "SSVI finite candidate (" + str(surface_fit.get("status", "unknown")) + ")"


def _prediction_metrics(rows):
    """Recompute price-space diagnostics with each fixed input uncertainty."""
    price_errors, weighted_errors, iv_errors = [], [], []
    for row in rows:
        if _finite(row.get("model_price")) and _finite(row.get("close")):
            error = row["model_price"] - row["close"]
            price_errors.append(error)
            if _finite(row.get("uncertainty_price")) and row["uncertainty_price"] > 0:
                weighted_errors.append(error / row["uncertainty_price"])
        if _finite(row.get("model_iv")) and _finite(row.get("iv")):
            iv_errors.append((row["model_iv"] - row["iv"]) * 10_000)

    def mean(values):
        return sum(values) / len(values) if values else None

    def rmse(values):
        return math.sqrt(sum(value * value for value in values) / len(values)) if values else None

    return {"target_count": len(rows), "price_comparable_count": len(price_errors),
            "price_rmse": rmse(price_errors), "mean_price_error": mean(price_errors),
            "max_abs_price_error": max(map(abs, price_errors), default=None),
            "weighted_comparable_count": len(weighted_errors), "weighted_rmse": rmse(weighted_errors),
            "mean_soft_l1": mean([math.hypot(1, error) - 1 for error in weighted_errors]),
            "iv_comparable_count": len(iv_errors), "iv_rmse_bp": rmse(iv_errors),
            "mean_abs_iv_error_bp": mean(list(map(abs, iv_errors))), "mean_iv_error_bp": mean(iv_errors)}


def _grouped_surface_metrics(surface_fit, calibration):
    """Avoid double-counting call/put pairs; keep split/exclusion roles apart."""
    bins = [0.9, 0.95, 1.0, 1.05, 1.1]

    def band(value):
        lower = 0.0
        for upper in bins:
            if value < upper:
                return f"[{lower:g}, {upper:g})"
            lower = upper
        return f"[{lower:g}, infinity)"

    grouped = defaultdict(list)
    datasets = [("SSVI", surface_fit.get("predictions", []))]
    if calibration:
        datasets.append(("Heston", calibration.get("predictions", [])))
    for model, predictions in datasets:
        for row in representatives(predictions):
            role = row.get("calibration_role", row.get("split", "unknown"))
            values = {"expiry": row["expiry"], "strike": row["strike"],
                      "moneyness": band(row["strike"] / row["forward"])}
            for dimension, value in values.items():
                grouped[(model, role, dimension, value)].append(row)
    rows = []
    for (model, split, dimension, value), selected in sorted(grouped.items()):
        rows.append({"model": model, "split": split, "dimension": dimension, "value": value,
                     **_prediction_metrics(selected),
                     "uncertainty_sources": sorted({str(row.get("uncertainty_source", "unknown")) for row in selected}),
                     "query_error_count": sum(bool(row.get("query_error")) for row in selected)})
    return {"basis": "one existing OTM representative per expiry/strike group; train, holdout and excluded expiries remain separate",
            "moneyness": "strike / forward; left-closed, right-open bands", "moneyness_edges": bins,
            "loss": "mean(sqrt(1 + ((model_price - Close) / fixed_uncertainty_price)^2) - 1)",
            "rows": rows}


def _coordinate_residual_plot(predictions, baseline, dimension, output, label, plt):
    expiries = sorted({row["expiry"] for row in predictions})
    expiry_index = {expiry: index for index, expiry in enumerate(expiries)}

    def coordinate(row):
        return row["strike"] if dimension == "strike" else expiry_index[row["expiry"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split, marker in (("train", "o"), ("holdout", "s"), ("excluded_expiry", "D")):
        selected = [row for row in predictions if row.get("split") == split]
        paired = [baseline[row["group"]] for row in selected if row["group"] in baseline]
        for model, group, shape, alpha in (("SSVI", selected, marker, 0.8), ("Heston", paired, "x", 0.6)):
            for ax, key in ((axes[0], "price_error"), (axes[1], "iv_error_bp")):
                finite = [row for row in group if _finite(row.get(key))]
                if finite:
                    ax.scatter([coordinate(row) for row in finite], [row[key] for row in finite],
                               marker=shape, alpha=alpha, label=model + " " + split)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Strike (index points)" if dimension == "strike" else "Contract expiry")
        if dimension == "expiry":
            ax.set_xticks(range(len(expiries)), expiries, rotation=35, ha="right", fontsize=8)
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("Model minus Close (index points)")
    axes[1].set_ylabel("Model minus Close IV (basis points)")
    fig.suptitle(label + "; OTM representatives by " + dimension)
    fig.tight_layout()
    name = "surface_residuals_by_" + dimension + ".png"
    fig.savefig(output / name, dpi=160)
    plt.close(fig)
    return name


def _surface_plots(validated, calibration, surface_fit, output, plt, np):
    """Show local surface candidates separately from supplier volatility."""
    predictions = representatives(surface_fit.get("predictions", []))
    rows = [row for row in predictions if _finite(row.get("model_iv"))]
    if not rows:
        return [], ["Requested SSVI fit has no finite predictions for plotting; inspect surface_fit.json."]
    label = _surface_label(surface_fit)
    baseline = {row["group"]: row for row in representatives(calibration.get("predictions", []))} if calibration else {}
    qualifying = set(surface_fit.get("qualifying_expiries", []))
    shown_expiries = sorted({row["expiry"] for row in predictions
                            if row["expiry"] in qualifying or (not qualifying and
                                row.get("calibration_role", row.get("split")) in {"train", "holdout"})})
    # Every fitted maturity belongs in this comparison. Sampling from all
    # predictions let a thin excluded maturity displace a fitted one.
    # Exclusions remain in residual charts, CSVs, and grouped diagnostics.
    artifacts, warnings = [], []
    if surface_fit.get("status") != "converged":
        warnings.append("SSVI did not converge: finite candidate charts are diagnostic and do not establish a successful surface fit.")
    fig, axes = plt.subplots(len(shown_expiries), 1, figsize=(10, 3.2 * len(shown_expiries)), squeeze=False)
    for ax, expiry in zip(axes[:, 0], shown_expiries):
        selected = sorted([row for row in predictions if row["expiry"] == expiry], key=lambda row: row["strike"])
        for split, marker in (("train", "o"), ("holdout", "s"), ("excluded_expiry", "D")):
            group = [row for row in selected if row.get("split") == split]
            if group:
                ax.scatter([row["strike"] / row["forward"] for row in group], [100 * row["iv"] for row in group],
                           marker=marker, label="Close IV " + split)
        supported = [row for row in selected if _finite(row.get("model_iv"))]
        if supported:
            ax.plot([row["strike"] / row["forward"] for row in supported], [100 * row["model_iv"] for row in supported], "-", label=label)
        heston = [baseline[row["group"]] for row in selected if row["group"] in baseline and _finite(baseline[row["group"]].get("model_iv"))]
        if heston:
            heston_label = "Heston baseline" if calibration.get("status") == "converged" else "Heston finite candidate (" + str(calibration.get("status")) + ")"
            ax.plot([row["strike"] / row["forward"] for row in heston], [100 * row["model_iv"] for row in heston], "--", label=heston_label)
        ax.set(title=expiry, xlabel="Strike / forward", ylabel="Black IV (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Local SSVI and independent Heston price fits; not supplier volatility", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output / "surface_smiles.png", dpi=160)
    plt.close(fig)
    artifacts.append("surface_smiles.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split, marker in (("train", "o"), ("holdout", "s"), ("excluded_expiry", "D")):
        selected = [row for row in predictions if row.get("split") == split]
        price_rows = [row for row in selected if _finite(row.get("price_error"))]
        iv_rows = [row for row in selected if _finite(row.get("iv_error_bp"))]
        if price_rows:
            axes[0].scatter([row["strike"] / row["forward"] for row in price_rows], [row["price_error"] for row in price_rows],
                            marker=marker, label="SSVI " + split)
        if iv_rows:
            axes[1].scatter([row["strike"] / row["forward"] for row in iv_rows], [row["iv_error_bp"] for row in iv_rows],
                            marker=marker, label="SSVI " + split)
        paired = [baseline[row["group"]] for row in selected if row["group"] in baseline and _finite(baseline[row["group"]].get("price_error"))]
        if paired:
            axes[0].scatter([row["strike"] / row["forward"] for row in paired], [row["price_error"] for row in paired],
                            marker="x", alpha=0.6, label="Heston " + split)
            paired_iv = [row for row in paired if _finite(row.get("iv_error_bp"))]
            axes[1].scatter([row["strike"] / row["forward"] for row in paired_iv], [row["iv_error_bp"] for row in paired_iv],
                            marker="x", alpha=0.6, label="Heston " + split)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Strike / forward")
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("Model minus Close (index points)")
    axes[1].set_ylabel("Model minus Close IV (basis points)")
    fig.suptitle(label + "; grouped strike holdout remains separate")
    fig.tight_layout()
    fig.savefig(output / "surface_residuals.png", dpi=160)
    plt.close(fig)
    artifacts.append("surface_residuals.png")
    for dimension in ("expiry", "strike"):
        artifacts.append(_coordinate_residual_plot(predictions, baseline, dimension, output, label, plt))

    # Evaluate the saved SSVI object itself, rather than drawing another
    # interpolation of IV predictions. Unsupported hull/curve queries stay
    # masked and are never replaced with wing or maturity extrapolation.
    if surface_fit.get("surface"):
        from .surface import VolSurface
        try:
            surface = VolSurface.from_dict(surface_fit["surface"])
            k_values = np.linspace(min(math.log(row["strike"] / row["forward"]) for row in rows),
                                   max(math.log(row["strike"] / row["forward"]) for row in rows), 60)
            t_min, t_max = min(row["t"] for row in rows), max(row["t"] for row in rows)
            t_values = np.unique(np.r_[np.linspace(t_min, t_max, 45),
                                       [node["t"] for node in surface.nodes if t_min <= node["t"] <= t_max]])
            log_grid, time_grid = np.meshgrid(k_values, t_values)
            iv_grid = np.full(log_grid.shape, np.nan)
            density_grid = np.full(log_grid.shape, np.nan)
            calendar_grid = np.full(log_grid.shape, np.nan)
            for index in np.ndindex(log_grid.shape):
                k, t = float(log_grid[index]), float(time_grid[index])
                try:
                    w = surface.total_variance(k, t)
                    derivative = surface.derivatives(k, t)
                    wk, wkk = derivative["w_k"], derivative["w_kk"]
                    iv_grid[index] = math.sqrt(w / t)
                    density_grid[index] = (1 - k * wk / (2 * w)) ** 2 - wk * wk / 4 * (1 / w + 0.25) + wkk / 2
                    calendar = [derivative.get(key) for key in ("w_t", "w_t_left", "w_t_right")]
                    finite_calendar = [value for value in calendar if _finite(value)]
                    if finite_calendar:
                        calendar_grid[index] = min(finite_calendar)
                except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
                    continue
            if np.any(np.isfinite(iv_grid)):
                fig = plt.figure(figsize=(10, 5.5))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(np.exp(log_grid), time_grid, 100 * iv_grid, cmap="viridis", linewidth=0, alpha=0.9)
                ax.set(xlabel="Strike / forward", ylabel="Maturity (years)", zlabel="Black IV (%)", title=label)
                fig.suptitle("Saved SSVI evaluation inside declared coverage; unsupported queries masked")
                fig.tight_layout()
                fig.savefig(output / "surface_model.png", dpi=160)
                plt.close(fig)
                artifacts.append("surface_model.png")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                for ax, grid, title in ((axes[0], density_grid, "Density factor g(k,T)"),
                                         (axes[1], calendar_grid, "Minimum one-sided dw/dT")):
                    if np.any(np.isfinite(grid)):
                        im = ax.pcolormesh(log_grid, time_grid, np.ma.masked_invalid(grid), shading="auto", cmap="viridis")
                        fig.colorbar(im, ax=ax)
                    ax.set(xlabel="Log(strike / forward)", ylabel="Maturity (years)", title=title)
                fig.suptitle("Finite-grid diagnostics; structural sufficient conditions are reported separately")
                fig.tight_layout()
                fig.savefig(output / "surface_constraints.png", dpi=160)
                plt.close(fig)
                artifacts.append("surface_constraints.png")
            else:
                warnings.append("The SSVI plotting grid has no supported finite queries inside its declared coverage.")
        except (ValueError, RuntimeError, KeyError) as exc:
            warnings.append("Saved SSVI candidate could not be evaluated for surface/constraint charts: " + str(exc))
    return artifacts, warnings


def _plots(validated, calibration, output, *, surface_fit=None):
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
    if not vendor_metadata:
        capture = validated.get("capture_diagnostics", {})
        verification = capture.get("vendor_surface_verification") or {}
        if verification.get("status") in {"unverified", "unavailable", "error"}:
            warnings.append("Supplier volatility overlay is unavailable: typed unit/model verification did not succeed. The untyped IPA matrix remains raw evidence and is excluded from comparisons; inspect capture_diagnostics for provider errors.")
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
    if surface_fit is not None:
        surface_images, surface_warnings = _surface_plots(validated, calibration, surface_fit, output, plt, np)
        artifacts.extend(surface_images)
        warnings.extend(surface_warnings)
    return artifacts, warnings


def _surface_comparison(surface_fit, calibration):
    """Check shared target inputs and recompute a common robust price loss."""
    surface_rows = representatives(surface_fit.get("predictions", []))
    heston_rows = representatives(calibration.get("predictions", [])) if calibration else []
    comparison = {"meaning": "Local SSVI and analytic Heston fits to observed Close prices; neither is vendor IV.",
                  "baseline_status": calibration.get("status") if calibration else "not_requested",
                  "common_objective": "mean(sqrt(1 + ((model_price - Close) / fixed_uncertainty_price)^2) - 1)",
                  "objective_normalization": "SSVI records the mean soft_l1 loss; SciPy Heston cost is the corresponding sum. A positive constant normalization does not change its minimum.",
                  "input_checks": ["ric", "expiry", "strike", "kind", "as_of", "spot", "close", "forward",
                                   "discount_factor", "t", "r", "q", "uncertainty_price", "uncertainty_source"]}
    for split, metric_key in (("train", "train"), ("holdout", "holdout")):
        surface_by_group = {row["group"]: row for row in surface_rows if row.get("split") == split}
        heston_by_group = {row["group"]: row for row in heston_rows if row.get("split") == split}
        surface_groups, heston_groups = set(surface_by_group), set(heston_by_group)
        shared = sorted(surface_groups & heston_groups)
        mismatches = []
        for group in shared:
            differing = [key for key in comparison["input_checks"]
                         if surface_by_group[group].get(key) != heston_by_group[group].get(key)]
            if differing:
                mismatches.append({"group": group, "fields": differing})
        same_groups = surface_groups == heston_groups if calibration else None
        comparison[split] = {"ssvi": surface_fit.get(metric_key),
                             "heston": calibration.get(metric_key) if calibration else None,
                             "same_target_groups": same_groups,
                             "same_target_inputs": bool(shared) and same_groups and not mismatches if calibration else None,
                             "input_mismatches": mismatches,
                             "ssvi_target_count": len(surface_groups), "heston_target_count": len(heston_groups),
                             "shared_target_count": len(shared),
                             "shared_target_metrics": {"SSVI": _prediction_metrics([surface_by_group[group] for group in shared]),
                                                       "Heston": _prediction_metrics([heston_by_group[group] for group in shared])} if calibration else None,
                             "ssvi_only_groups": sorted(surface_groups - heston_groups),
                             "heston_only_groups": sorted(heston_groups - surface_groups)}
    return comparison


def _surface_html(surface_fit, comparison, grouped):
    def evidence(title, explanation, value):
        payload = html.escape(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        return f"<h3>{html.escape(title)}</h3><p>{html.escape(explanation)}</p><pre>{payload}</pre>"

    status = str(surface_fit.get("status", "unknown"))
    result = '<h2>Local SSVI surface fit</h2><p class="note">'
    result += html.escape("Status: " + status + ". This is a locally fitted Black implied-volatility surface, not supplier volatility or an independent market quote. ")
    if status != "converged":
        result += "Finite candidates are diagnostic; this requested fit did not complete successfully. "
    result += "Structural constraints and finite-grid diagnostics have separate meanings; inspect the declared coverage and unsupported queries.</p>"
    result += evidence("Training fit and strike holdout",
                       "Start selection uses training data only. Every-fifth-strike holdout measures within-expiry interpolation, not independent expiry prediction.",
                       {"train": surface_fit.get("train"), "holdout": surface_fit.get("holdout"),
                        "qualifying_expiries": surface_fit.get("qualifying_expiries"),
                        "excluded_expiries": surface_fit.get("excluded_expiries"),
                        "parameters": surface_fit.get("parameters"), "settings": surface_fit.get("settings")})
    result += evidence("Heston baseline on the same targets",
                       "Target groups, Close/carry inputs and fixed uncertainties are checked before comparing a common normalized robust loss. No Monte Carlo result is used as the surface calibration objective.", comparison)
    expiry_metrics = [row for row in grouped["rows"] if row["dimension"] == "expiry"]
    result += evidence("Price and IV residuals by expiry, strike, and moneyness",
                       "The expiry summary below separates train/holdout/excluded roles and one OTM representative per group. Full strike and explicit moneyness-band summaries are in surface_grouped_residuals.csv and surface_diagnostics.json; unpriceable targets retain their counts.",
                       {"basis": grouped["basis"], "moneyness_edges": grouped["moneyness_edges"], "expiry_metrics": expiry_metrics})
    diagnostic = surface_fit.get("diagnostics") or {}
    result += evidence("Structural constraints and grid diagnostics",
                       "Numerical density/calendar scans are evidence on their recorded grid; they are not a substitute for the structural sufficient conditions.",
                       {key: diagnostic.get(key) for key in ("static_arbitrage", "coverage", "interpolation", "meaning")})
    settings = surface_fit.get("settings") or {}
    result += evidence("Initialisation, bounds, and parameter dispersion",
                       "Bounds and sufficient-condition margins are distinct from optimizer convergence. Dispersion covers feasible starts, including ATM total-variance nodes, and is not a confidence interval.",
                       {"initialization": diagnostic.get("initialization"),
                        "bounds": {key: settings.get(key) for key in ("rho_bounds", "eta_bounds", "theta_bounds", "epsilon")},
                        "bound_activity": diagnostic.get("bound_activity"),
                        "parameter_dispersion": diagnostic.get("parameter_dispersion"),
                        "dispersion_meaning": diagnostic.get("parameter_dispersion_meaning")})
    result += evidence("Multiple optimisation starts",
                       "Every start and its feasibility/convergence outcome is retained. The chosen start is selected without consulting strike or expiry holdouts.",
                       {"chosen_start": surface_fit.get("chosen_start"), "starts": surface_fit.get("starts")})
    result += evidence("Expiry holdout folds",
                       "Each fold must refit without the held-out expiry. Unsupported queries and missing folds remain explicit; this is separate from the strike holdout.",
                       surface_fit.get("expiry_holdouts"))
    result += evidence("Input and fit sensitivities",
                       "Sensitivity refits are diagnostic assumptions and report their own status. They do not overwrite the selected main surface or prove parameter identification.",
                       surface_fit.get("sensitivities"))
    return result


def write_report(validated, calibration_or_none, numerical_or_none, output, *, surface_fit=None):
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
    numerical_records = numerical_or_none.get("records", []) if numerical_or_none else []
    numerical_fields += sorted({key for row in numerical_records for key in row
                                if key not in numerical_fields})
    _write_csv(output / "numerical_validation.csv", numerical_records, numerical_fields)
    surface_links = []
    surface_comparison = None
    surface_grouped = None
    if surface_fit is not None:
        _write_json(output / "surface.json", surface_fit.get("surface"))
        _write_json(output / "surface_fit.json", surface_fit)
        surface_comparison = _surface_comparison(surface_fit, calibration_or_none)
        surface_grouped = _grouped_surface_metrics(surface_fit, calibration_or_none)
        _write_json(output / "surface_diagnostics.json",
                    {"schema_version": 1, "status": surface_fit.get("status"),
                     "diagnostics": surface_fit.get("diagnostics"),
                     "shared_target_comparison": surface_comparison, "grouped_residuals": surface_grouped,
                     "expiry_holdouts": surface_fit.get("expiry_holdouts"),
                     "sensitivities": surface_fit.get("sensitivities")})
        surface_rows = [{**row, "moneyness": row["strike"] / row["forward"],
                         "log_forward_moneyness": math.log(row["strike"] / row["forward"])}
                        for row in surface_fit.get("predictions", [])]
        surface_fields = ["ric", "expiry", "strike", "kind", "close", "iv", "model_price", "price_error",
                          "model_iv", "iv_error_bp", "iv_error_reason", "query_error", "uncertainty_price",
                          "uncertainty_source", "weighted_error", "split", "calibration_role", "group",
                          "moneyness", "log_forward_moneyness", "forward", "discount_factor", "t", "r", "q"]
        surface_fields += sorted({key for row in surface_rows for key in row if key not in surface_fields})
        _write_csv(output / "surface_residuals.csv", surface_rows, surface_fields)
        grouped_fields = ["model", "split", "dimension", "value", "target_count", "price_comparable_count",
                          "price_rmse", "mean_price_error", "max_abs_price_error", "weighted_comparable_count",
                          "weighted_rmse", "mean_soft_l1", "iv_comparable_count", "iv_rmse_bp",
                          "mean_abs_iv_error_bp", "mean_iv_error_bp", "uncertainty_sources", "query_error_count"]
        _write_csv(output / "surface_grouped_residuals.csv", surface_grouped["rows"], grouped_fields)
        surface_links = ["surface.json", "surface_fit.json", "surface_diagnostics.json", "surface_residuals.csv",
                         "surface_grouped_residuals.csv"]
    images, warnings = _plots(validated, calibration_or_none, output, surface_fit=surface_fit)
    if surface_fit is not None and surface_fit.get("status") != "converged":
        warnings.append("Requested SSVI fit status is " + str(surface_fit.get("status", "unknown")) + "; candidate artifacts must not be treated as a converged surface.")
    if calibration_or_none and surface_comparison is not None:
        for split in ("train", "holdout"):
            if not surface_comparison[split]["same_target_inputs"]:
                warnings.append("SSVI/Heston " + split + " target groups or Close/carry/uncertainty inputs differ; these metrics are not a comparison on identical inputs.")
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
    summary["capture_diagnostics"] = validated.get("capture_diagnostics", {})
    if surface_fit is not None:
        summary.update(surface_status=surface_fit.get("status"), surface_model="ssvi",
                       surface_parameters=surface_fit.get("parameters"), surface_settings=surface_fit.get("settings"),
                       surface_train=surface_fit.get("train"), surface_strike_holdout=surface_fit.get("holdout"),
                       surface_comparison=surface_comparison, surface_chosen_start=surface_fit.get("chosen_start"),
                       surface_qualifying_expiries=surface_fit.get("qualifying_expiries"),
                       surface_excluded_expiries=surface_fit.get("excluded_expiries"))
    _write_json(output / "summary.json", summary)
    links = ["summary.json", "validation.json", "calibration.json", "numerical_validation.json", "residuals.csv", "numerical_validation.csv"]
    links.extend(surface_links)
    figures = "\n".join(f'<figure><img src="{html.escape(name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in images)
    link_html = " · ".join(f'<a href="{html.escape(name)}">{html.escape(name)}</a>' for name in links)
    checks = numerical_or_none.get("summaries", []) if numerical_or_none else []
    check_rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
        row["ric"], row["model"], row.get("estimator", "plain"),
        format(row["mean_price"], ".9g"), format(row["reference_price"], ".9g"),
        format(row["combined_std_error"], ".3g"),
        format(100 * row["relative_std_error"], ".3g") if row.get("relative_std_error") is not None else "—",
        row["classification"])) + "</tr>" for row in checks)
    check_table = ("<h2>Independent numerical checks</h2><table><thead><tr>"
                   "<th>Contract</th><th>Model</th><th>Estimator</th><th>Price</th>"
                   "<th>Reference</th><th>Standard error</th><th>SE / reference (%)</th><th>Outcome</th>"
                   "</tr></thead><tbody>" + check_rows + "</tbody></table>"
                   "<p>Original sampling results, rare-event rechecks, reference quadrature, "
                   "and grid changes are retained in the JSON and CSV evidence.</p>") if checks else ""
    surface_section = _surface_html(surface_fit, surface_comparison, surface_grouped) if surface_fit is not None else ""
    surface_note = "Surface plots interpolate total variance inside the observed hull, without extrapolation or an arbitrage-free claim."
    if surface_fit is not None:
        surface_note = ("Market-knot plots interpolate total variance inside the observed hull without an arbitrage-free claim. "
                        "Separate SSVI charts evaluate the saved constrained model; its structural conditions, coverage, and finite-grid diagnostics are reported below.")
    page = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPX EOD market validation</title><style>
body{{font:16px/1.5 system-ui,sans-serif;color:#17212b;max-width:1200px;margin:40px auto;padding:0 24px}}
pre{{background:#f3f5f7;padding:20px;overflow:auto}}img{{max-width:100%;height:auto}}figure{{margin:32px 0}}
a{{color:#075b94}}.note{{border-left:4px solid #d49721;padding:12px 18px;background:#fff8e6}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #dce2e8;padding:8px;text-align:left}}
</style><h1>SPX EOD market validation</h1>
<p class="note">Observed references are Close prices, not bid/ask mid prices. Missing historical bid/ask does not imply a tradable price interval.
Calibration residuals measure market fit; independent core-versus-QuantLib results measure sampling and discretization or implementation error.
The every-fifth-strike holdout tests interpolation within observed expiries. {surface_note}</p>
<p>{link_html}</p>{check_table}{surface_section}<h2>Run summary</h2><pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))}</pre>
{figures}</html>'''
    path = output / "report.html"
    path.write_text(page, encoding="utf-8")
    return path
