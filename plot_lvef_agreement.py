"""Create publication-ready correlation and Bland-Altman plots from patient LVEF pairs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lvef_metrics import calculate_lvef_metrics


def read_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    reference, prediction, patients = [], [], set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"patient", "gt_ef", "pred_ef"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: patient, gt_ef, pred_ef")
        for line, row in enumerate(reader, 2):
            patient = (row.get("patient") or "").strip()
            if not patient or patient in patients:
                raise ValueError(f"{path}:{line}: missing or duplicate patient {patient!r}")
            gt, pred = float(row["gt_ef"]), float(row["pred_ef"])
            if not np.isfinite([gt, pred]).all():
                raise ValueError(f"{path}:{line}: non-finite LVEF value")
            patients.add(patient)
            reference.append(gt)
            prediction.append(pred)
    if len(reference) < 3:
        raise ValueError("At least three valid patient pairs are required")
    return np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float)


def invalid_count(csv_path: Path) -> int:
    path = csv_path.with_name("clinical_invalid.csv")
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def axis_limits(*arrays: np.ndarray) -> tuple[float, float]:
    low = min(float(np.min(values)) for values in arrays)
    high = max(float(np.max(values)) for values in arrays)
    padding = max(2.0, 0.06 * (high - low or 1.0))
    return max(0.0, np.floor(low - padding)), min(100.0, np.ceil(high + padding))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot patient-level LVEF correlation and Bland-Altman agreement."
    )
    parser.add_argument("--csv", type=Path, required=True,
                        help="clinical_per_patient.csv containing patient, gt_ef and pred_ef")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    output = (args.output_dir or csv_path.parent).resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference, prediction = read_pairs(csv_path)
    invalid_n = invalid_count(csv_path)
    summary = calculate_lvef_metrics(reference, prediction, invalid_n=invalid_n)
    means = (reference + prediction) / 2.0
    differences = prediction - reference

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)

    # Correlation panel: identity line describes agreement; regression line describes association.
    ax = axes[0]
    lower, upper = axis_limits(reference, prediction)
    fit = np.polyfit(reference, prediction, 1)
    x_fit = np.linspace(lower, upper, 200)
    ax.scatter(reference, prediction, s=24, color="#2878B5", edgecolor="white",
               linewidth=0.45, alpha=0.88, zorder=3)
    ax.plot([lower, upper], [lower, upper], color="#555555", linestyle="--",
            linewidth=1.0, label="Identity line")
    ax.plot(x_fit, fit[0] * x_fit + fit[1], color="#C82423", linewidth=1.2,
            label="Linear fit")
    ax.set(xlim=(lower, upper), ylim=(lower, upper),
           xlabel="Reference LVEF (%)", ylabel="Predicted LVEF (%)",
           title="(a) Correlation analysis")
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.04, 0.96,
            f"r = {summary['corr']:.3f}\nMAE = {summary['mae']:.2f} pp\nn = {summary['n']}",
            transform=ax.transAxes, va="top", ha="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    # Bland-Altman panel: differences are always prediction minus reference.
    ax = axes[1]
    bias = float(summary["bias"])
    loa_lower = float(summary["loa_lower"])
    loa_upper = float(summary["loa_upper"])
    ax.scatter(means, differences, s=24, color="#2878B5", edgecolor="white",
               linewidth=0.45, alpha=0.88, zorder=3)
    ax.axhline(bias, color="#C82423", linewidth=1.2, label=f"Bias = {bias:.2f} pp")
    ax.axhline(loa_upper, color="#555555", linestyle="--", linewidth=1.0,
               label=f"Upper LoA = {loa_upper:.2f} pp")
    ax.axhline(loa_lower, color="#555555", linestyle="--", linewidth=1.0,
               label=f"Lower LoA = {loa_lower:.2f} pp")
    diff_padding = max(2.0, 0.10 * (float(differences.max()) - float(differences.min()) or 1.0))
    ax.set_ylim(min(float(differences.min()), loa_lower) - diff_padding,
                max(float(differences.max()), loa_upper) + diff_padding)
    ax.set(xlabel="Mean of reference and predicted LVEF (%)",
           ylabel="Difference (predicted - reference) (pp)",
           title="(b) Bland-Altman analysis")
    ax.legend(loc="best", frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)

    png = output / "SIMSAM_LVEF_correlation_BlandAltman.png"
    pdf = output / "SIMSAM_LVEF_correlation_BlandAltman.pdf"
    figure.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    figure_summary = {
        "source_csv": str(csv_path),
        "error_direction": "prediction - reference",
        "unit": "percentage_points",
        "n": summary["n"],
        "invalid_n": summary["invalid_n"],
        "pearson_r": summary["corr"],
        "pearson_r_percent": summary["corr_percent"],
        "mae": summary["mae"],
        "bias": summary["bias"],
        "loa_lower": summary["loa_lower"],
        "loa_upper": summary["loa_upper"],
    }
    with (output / "SIMSAM_LVEF_figure_values.json").open("w", encoding="utf-8") as stream:
        json.dump(figure_summary, stream, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"[DONE] PNG: {png}")
    print(f"[DONE] PDF: {pdf}")
    print(json.dumps(figure_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
