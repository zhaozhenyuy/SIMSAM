from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from lvef_metrics import calculate_lvef_metrics, format_lvef_summary


MODEL_NAMES = {
    "unet-master": "U-Net", "Swin-Unet": "SwinUNet", "H2Former-main": "H2Former",
    "MedSAM": "MedSAM", "Medical-SAM-Adapter": "MSA", "SAMed": "SAMed", "SAMUS": "SAMUS",
}


def read_pairs(path: Path, unit: str = "percent") -> list[dict]:
    """Never infer units, drop bad records, or trust legacy error/bias columns."""
    rows = []
    patients = set()
    scale = 100.0 if unit == "fraction" else 1.0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"patient", "gt_ef", "pred_ef"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: require patient,gt_ef,pred_ef columns")
        for line, row in enumerate(reader, 2):
            patient = (row.get("patient") or "").strip()
            if not patient or patient in patients:
                raise ValueError(f"{path}:{line}: missing or duplicate patient {patient!r}")
            try:
                gt, pred = float(row["gt_ef"]) * scale, float(row["pred_ef"]) * scale
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: invalid LVEF value") from error
            if not np.isfinite([gt, pred]).all():
                raise ValueError(f"{path}:{line}: non-finite LVEF pair")
            patients.add(patient)
            rows.append({
                "patient": patient, "gt_ef": gt, "pred_ef": pred,
                "pair_mean": (pred + gt) / 2.0,
                "error": pred - gt, "absolute_error": abs(pred - gt),
            })
    if not rows:
        raise ValueError(f"{path}: no patient pairs")
    return rows


def invalid_count(path: Path) -> int | None:
    invalid_path = path.with_name("clinical_invalid.csv")
    if invalid_path.is_file():
        with invalid_path.open(newline="", encoding="utf-8-sig") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    summary_path = path.with_name("clinical_summary.json")
    if summary_path.is_file():
        with summary_path.open(encoding="utf-8-sig") as stream:
            count = json.load(stream).get("invalid_n")
        return int(count) if count is not None else None
    return None


def evaluate_csv(path: Path, unit: str, repeats: int, seed: int):
    rows = read_pairs(path, unit)
    summary = calculate_lvef_metrics(
        [row["gt_ef"] for row in rows], [row["pred_ef"] for row in rows],
        invalid_n=invalid_count(path), repeats=repeats, seed=seed,
    )
    cohort = sorted((row["patient"], row["gt_ef"]) for row in rows)
    summary["source_csv"] = str(path.resolve())
    summary["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary["cohort_sha256"] = hashlib.sha256(json.dumps(cohort).encode()).hexdigest()
    summary["input_unit"] = unit
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Recompute LVEF corr, signed Bias, MAE and 95% LoA from patient pairs; no inference or PSD.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="Patient CSV with patient,gt_ef,pred_ef")
    source.add_argument("--scan-root", type=Path, help="Find all clinical_per_patient.csv recursively; keep runs separate")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory; originals are never overwritten")
    parser.add_argument("--unit", choices=("percent", "fraction"), default="percent", help="percent: 60 means 60%%; fraction: 0.60 means 60%%")
    parser.add_argument("--model", default="", help="Optional name for --csv")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be a new or empty directory: {output}")
    if args.scan_root:
        root = args.scan_root.resolve()
        if not root.is_dir():
            parser.error(f"Scan root not found: {root}")
        paths = sorted(path for path in root.rglob("clinical_per_patient.csv") if output not in path.parents)
    else:
        paths = [args.csv.resolve()]
    if not paths:
        parser.error("No clinical_per_patient.csv found")

    # Validate every source before writing any output.
    prepared = [(path, *evaluate_csv(path, args.unit, args.bootstrap_repeats, args.seed)) for path in paths]
    output.mkdir(parents=True, exist_ok=True)
    combined = []
    for path, rows, summary in prepared:
        relative = path.parent.relative_to(root) if args.scan_root else Path(".")
        target = output / relative
        target.mkdir(parents=True, exist_ok=True)
        model = args.model or next((MODEL_NAMES[part] for part in path.parts if part in MODEL_NAMES), path.parent.name)
        summary["model"] = model
        summary["run"] = path.parent.name
        with (target / "lvef_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        write_csv(target / "lvef_pairs.csv", rows)
        keys = ("n", "invalid_n", "corr_percent", "corr_bootstrap_mean_percent", "corr_bootstrap_std_percent", "bias", "mae", "loa_lower", "loa_upper", "error_sd", "cohort_sha256", "source_csv")
        combined.append({"model": model, "run": path.parent.name, **{key: summary[key] for key in keys}})
        print(f"\n[{model}] {path.parent.name}\n{format_lvef_summary(summary)}")
    write_csv(output / "lvef_results.csv", combined)
    if len({row["cohort_sha256"] for row in combined}) > 1:
        print("[WARN] Patient cohorts/reference values differ. Do not rank these runs as a matched comparison.")
    print(f"\n[DONE] {len(combined)} runs, kept separate. Results: {output}")


if __name__ == "__main__":
    main()
