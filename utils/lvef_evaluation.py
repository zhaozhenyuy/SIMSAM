from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from lvef_metrics import calculate_lvef_metrics, format_lvef_summary


def validate_clinical_request(args, opt):
    if not getattr(args, "compute_ef", False):
        return
    if getattr(opt, "eval_mode", "") != "camus":
        raise ValueError(
            "--compute_ef uses CAMUS paired 2CH/4CH ED/ES masks. "
            "Use --task CAMUS_Video_Full; Echo single-view masks cannot use this biplane protocol."
        )
    if int(opt.batch_size) != 1:
        raise ValueError("This video LVEF evaluator requires --batch_size 1 --n_gpu 1")


def record_camus_endpoints(mask_dict, gt_efs, names, reference_ef, spacing, masks):
    """Match by patient/view, not the incidental insertion order of dictionaries."""
    if len(names) != 1 or len(masks) != 1:
        raise ValueError("CAMUS video LVEF evaluation requires a batch size of one")
    name = Path(str(names[0])).stem
    identity = re.fullmatch(r"(patient\d+)_([24]CH)", name, flags=re.I)
    if identity is None:
        raise ValueError(f"Expected CAMUS patient/view name, got {name!r}")
    patient, view = identity.group(1).lower(), identity.group(2).upper()
    reference = float(np.asarray(reference_ef).reshape(-1)[0])
    if patient in gt_efs and not np.isclose(gt_efs[patient], reference, atol=1e-3, rtol=0):
        raise ValueError(f"Inconsistent 2CH/4CH reference EF for {patient}")
    views = mask_dict.setdefault(patient, {})
    if view in views:
        raise ValueError(f"Duplicate CAMUS patient/view: {patient}_{view}")
    gt_efs[patient] = reference
    views[view] = {"ED": masks[0, 0], "ES": masks[0, -1], "spacing": spacing}


def save_camus_lvef(mask_dict, gt_efs, args, opt, compute_volumes):
    validate_clinical_request(args, opt)
    specified = getattr(args, "clinical_output_dir", "")
    if specified:
        output = Path(specified)
    else:
        base = Path(getattr(args, "test_log_dir", "") or opt.result_path)
        output = base / "lvef" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    filenames = ("clinical_per_patient.csv", "clinical_summary.json", "clinical_invalid.csv")
    if any((output / name).exists() for name in filenames):
        raise FileExistsError(f"Clinical results already exist; choose a new --clinical_output_dir: {output}")

    rows, invalid = [], []
    for patient in sorted(set(mask_dict) | set(gt_efs)):
        try:
            views = mask_dict[patient]
            for view in ("2CH", "4CH"):
                if view not in views:
                    raise ValueError(f"Missing {view} predictions")
                for endpoint in ("ED", "ES"):
                    if not np.asarray(views[view][endpoint]).any():
                        raise ValueError(f"Empty {view} {endpoint} predicted mask")
            edv, esv = compute_volumes(
                a2c_ed=views["2CH"]["ED"], a2c_es=views["2CH"]["ES"],
                a2c_voxelspacing=views["2CH"]["spacing"],
                a4c_ed=views["4CH"]["ED"], a4c_es=views["4CH"]["ES"],
                a4c_voxelspacing=views["4CH"]["spacing"],
            )
            swapped = bool(esv > edv)
            # Preserve the original CAMUS evaluator's EDV/ESV ordering safeguard.
            if swapped:
                edv, esv = esv, edv
            if not np.isfinite([edv, esv, gt_efs[patient]]).all() or edv <= 0:
                raise ValueError("Non-finite reference/volumes or non-positive EDV")
            pred = round(100.0 * (edv - esv) / edv, 2)
            gt = float(gt_efs[patient])
            rows.append({
                "patient": patient, "gt_ef": gt, "pred_edv": float(edv),
                "pred_esv": float(esv), "pred_ef": pred, "error": pred - gt,
                "absolute_error": abs(pred - gt), "pair_mean": (pred + gt) / 2.0,
                "volumes_swapped": swapped,
            })
        except (KeyError, IndexError, TypeError, ValueError, RuntimeError, ZeroDivisionError) as error:
            invalid.append({"patient": patient, "reason": str(error)})

    for name, records, fields in (
        ("clinical_per_patient.csv", rows, ("patient", "gt_ef", "pred_edv", "pred_esv", "pred_ef", "error", "absolute_error", "pair_mean", "volumes_swapped")),
        ("clinical_invalid.csv", invalid, ("patient", "reason")),
    ):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    if not rows:
        raise RuntimeError(f"No valid CAMUS LVEF pairs; see {output / 'clinical_invalid.csv'}")
    summary = calculate_lvef_metrics(
        [row["gt_ef"] for row in rows], [row["pred_ef"] for row in rows], invalid_n=len(invalid),
    )
    summary["protocol"] = {
        "task": getattr(args, "task", "CAMUS_Video_Full"),
        "model": getattr(args, "modelname", ""),
        "checkpoint": str(getattr(args, "load_path", "")),
        "enable_memory": getattr(args, "enable_memory", None),
        "reinforce": getattr(args, "reinforce", None),
        "disable_point_prompt": getattr(args, "disable_point_prompt", None),
        "disable_dino_prompt": getattr(args, "disable_dino_prompt", None),
        "enable_phase_memory": getattr(args, "enable_phase_memory", None),
        "enable_apfe": getattr(args, "enable_apfe", None),
        "data_path": str(opt.data_path),
        "split": getattr(args, "split", "") or opt.test_split,
        "volume_estimation": "CAMUS paired-view biplane Simpson",
        "component_policy": "largest",
        "component_connectivity": 4,
        "swap_if_esv_gt_edv": True, "round_pred_ef_decimals": 2,
        "swapped_n": sum(row["volumes_swapped"] for row in rows),
        "prediction_threshold": 0.6,
    }
    with (output / "clinical_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(format_lvef_summary(summary))
    if invalid:
        print(f"[WARN] {len(invalid)} invalid patients excluded; see clinical_invalid.csv before comparing methods.")
    print(f"[LVEF] Patient pairs and summary saved to: {output.resolve()}")
    return summary
