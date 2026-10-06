from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


SCHEMA_VERSION = "lvef_agreement_v2"


def _paired_values(reference: np.ndarray, prediction: np.ndarray):
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if reference.shape != prediction.shape or reference.size == 0:
        raise ValueError("LVEF metrics require nonempty, equally sized paired values")
    if not (np.isfinite(reference).all() and np.isfinite(prediction).all()):
        raise ValueError("LVEF pairs contain NaN or infinity; fix or explicitly record invalid patients")
    return reference, prediction


def bootstrap_correlation(
    reference: np.ndarray,
    prediction: np.ndarray,
    repeats: int = 2000,
    seed: int = 1234,
) -> Dict[str, float | int]:
    """Patient-level bootstrap; preserve the original evaluator's calculation."""
    reference, prediction = _paired_values(reference, prediction)
    if reference.size < 3:
        raise ValueError("Bootstrap correlation requires at least three paired values")
    if repeats < 2:
        raise ValueError("Bootstrap correlation requires at least two repeats")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, reference.size, size=(repeats, reference.size))
    sampled_reference = reference[indices]
    sampled_prediction = prediction[indices]
    reference_centered = sampled_reference - sampled_reference.mean(axis=1, keepdims=True)
    prediction_centered = sampled_prediction - sampled_prediction.mean(axis=1, keepdims=True)
    numerator = np.mean(reference_centered * prediction_centered, axis=1)
    denominator = sampled_reference.std(axis=1) * sampled_prediction.std(axis=1)
    valid = denominator > 0
    correlations = numerator[valid] / denominator[valid]
    if correlations.size < 2:
        raise RuntimeError("Too few valid bootstrap correlation samples")
    lower, upper = np.percentile(correlations, [2.5, 97.5])
    return {
        "repeats": int(repeats), "valid_repeats": int(correlations.size),
        "seed": int(seed), "mean": float(correlations.mean()),
        "std": float(correlations.std(ddof=1)),
        "ci95_lower": float(lower), "ci95_upper": float(upper),
    }


def calculate_lvef_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    invalid_n: int | None = 0,
    repeats: int = 2000,
    seed: int = 1234,
) -> Dict[str, object]:
    """Compute agreement from patient pairs expressed as percentages (e.g. 60)."""
    reference, prediction = _paired_values(reference, prediction)
    if repeats < 2:
        raise ValueError("Bootstrap correlation requires at least two repeats")
    if invalid_n is not None and invalid_n < 0:
        raise ValueError("invalid_n cannot be negative")
    errors = prediction - reference
    bias = float(errors.mean())
    error_sd = float(errors.std(ddof=1)) if errors.size > 1 else None
    denominator = float(reference.std(ddof=0) * prediction.std(ddof=0))
    corr = (
        float(np.mean((reference - reference.mean()) * (prediction - prediction.mean())) / denominator)
        if denominator > 0 else None
    )
    warnings = []
    if error_sd is None:
        warnings.append("At least two valid patients are needed for error SD and LoA.")
    if corr is None:
        warnings.append("Pearson correlation is undefined with a constant input or fewer than two pairs.")
    summary: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "n": int(reference.size), "invalid_n": invalid_n,
        "total_n": int(reference.size) + invalid_n if invalid_n is not None else None,
        "unit": "percentage_points", "error_direction": "prediction - reference",
        "corr": corr, "corr_percent": corr * 100.0 if corr is not None else None,
        "bias": bias, "mae": float(np.abs(errors).mean()),
        "error_sd": error_sd, "error_sd_ddof": 1,
        "loa_lower": bias - 1.96 * error_sd if error_sd is not None else None,
        "loa_upper": bias + 1.96 * error_sd if error_sd is not None else None,
        "loa_multiplier": 1.96,
        "corr_bootstrap_repeats": int(repeats), "corr_bootstrap_seed": int(seed),
        "corr_bootstrap_valid_repeats": 0,
        "corr_bootstrap_mean": None, "corr_bootstrap_std": None,
        "corr_bootstrap_mean_percent": None, "corr_bootstrap_std_percent": None,
        "corr_bootstrap_ci95_lower": None, "corr_bootstrap_ci95_upper": None,
        "definition": {
            "bias": "mean(predicted LVEF - reference LVEF); signed, not MAE",
            "mae": "mean(abs(predicted LVEF - reference LVEF))",
            "error_sd": "sample SD of paired differences (ddof=1); NOT prediction SD",
            "loa": "bias +/- 1.96 * sample SD of paired differences",
            "loa_interpretation": "Approximate 95% limits of agreement for independent patient differences with approximately normal, homoscedastic errors; not a confidence interval of the bias. These assumptions are not automatically verified.",
            "corr": "Pearson correlation of paired reference and predicted LVEF",
            "corr_bootstrap": "Patient-level bootstrap mean, sample SD and percentile 95% CI of Pearson correlation",
            "legacy_migration": "Old clinical_summary.json used bias for MAE and std for prediction SD. v2 uses signed bias and separate mae; prediction SD/variance are not reported.",
        },
        "warnings": warnings,
    }
    if reference.size >= 3 and corr is not None:
        try:
            boot = bootstrap_correlation(reference, prediction, repeats, seed)
            for key in ("mean", "std", "ci95_lower", "ci95_upper", "valid_repeats"):
                summary[f"corr_bootstrap_{key}"] = boot[key]
            summary["corr_bootstrap_mean_percent"] = float(boot["mean"]) * 100.0
            summary["corr_bootstrap_std_percent"] = float(boot["std"]) * 100.0
        except RuntimeError as error:
            warnings.append(str(error))
    else:
        warnings.append("Correlation bootstrap unavailable: need at least three pairs and nonconstant inputs.")
    return summary


def format_lvef_summary(summary: Mapping[str, object]) -> str:
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Legacy summary: recompute from patient pairs before reporting signed Bias")

    def number(key: str) -> str:
        value = summary.get(key)
        return "N/A" if value is None else f"{float(value):.6f}"

    return (
        "=== LVEF agreement (v2; errors in percentage points) ===\n"
        f"N={summary['n']} invalid={summary['invalid_n']} "
        f"corr={number('corr')} corr(%)={number('corr_percent')}\n"
        f"Bias(pred-ref)={number('bias')} MAE={number('mae')} "
        f"95%LoA=[{number('loa_lower')}, {number('loa_upper')}]\n"
        f"Difference_SD(ddof=1)={number('error_sd')}\n"
        f"corr_bootstrap(%)={number('corr_bootstrap_mean_percent')} +/- "
        f"{number('corr_bootstrap_std_percent')} "
        f"corr_95%CI=[{number('corr_bootstrap_ci95_lower')}, "
        f"{number('corr_bootstrap_ci95_upper')}] "
        f"repeats={summary['corr_bootstrap_valid_repeats']}/{summary['corr_bootstrap_repeats']}"
    )
