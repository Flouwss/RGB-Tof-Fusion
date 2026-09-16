from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import weighted_median as wmf_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused WMF parameter search against GT and nearest-neighbor ToF baseline."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--gt-dir", type=str, default="depthmaps")
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--tof-dir", type=str, default="tof_rect")
    parser.add_argument("--amplitude-dir", type=str, default="amplitude_rect")
    parser.add_argument("--baseline-dir", type=str, default="tof_rect_upsampled_nearest")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("wmf_parameter_search.json"),
    )
    return parser.parse_args()


def common_stems(*dirs: Path) -> list[str]:
    sets = [{p.stem for p in directory.glob("*.png")} for directory in dirs]
    return sorted(set.intersection(*sets), key=lambda name: int(name))


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)
    diff = pred_f - gt_f
    abs_diff = np.abs(diff)
    sq_diff = diff * diff
    ratio = np.maximum(
        np.clip(pred_f, 1e-6, None) / np.clip(gt_f, 1e-6, None),
        np.clip(gt_f, 1e-6, None) / np.clip(pred_f, 1e-6, None),
    )
    return {
        "mae": float(abs_diff.mean()),
        "rmse": float(np.sqrt(sq_diff.mean())),
        "abs_rel": float((abs_diff / np.clip(gt_f, 1e-6, None)).mean()),
        "delta_1.10": float((ratio < 1.10).mean()),
        "delta_1.25": float((ratio < 1.25).mean()),
    }


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0].keys()}


def evaluate_config(
    frames: list[dict[str, np.ndarray]],
    params: dict[str, float],
    wmf_module,
) -> dict[str, float]:
    full_rows: list[dict[str, float]] = []
    overlap_rows: list[dict[str, float]] = []

    for frame in frames:
        valid_mask = (frame["tof"] > 0) & (frame["amplitude"] >= params["min_amplitude"])
        filtered_small = wmf_module.joint_weighted_median_filter(
            tof=frame["tof"],
            rgb_guidance=frame["rgb_small"],
            valid_mask=valid_mask,
            radius=int(params["radius"]),
            sigma_spatial=float(params["sigma_spatial"]),
            sigma_color=float(params["sigma_color"]),
        )
        pred = cv2.resize(
            filtered_small,
            (frame["gt"].shape[1], frame["gt"].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        pred_valid = pred >= 1
        gt_valid = frame["gt"] >= 1
        base_valid = frame["baseline"] >= 1

        full_mask = pred_valid & gt_valid
        overlap_mask = full_mask & base_valid

        full_row = compute_metrics(pred[full_mask], frame["gt"][full_mask])
        full_row["coverage"] = float(full_mask.mean())
        full_rows.append(full_row)

        overlap_row = compute_metrics(pred[overlap_mask], frame["gt"][overlap_mask])
        overlap_row["coverage"] = float(overlap_mask.mean())
        overlap_rows.append(overlap_row)

    full = summarize_rows(full_rows)
    overlap = summarize_rows(overlap_rows)
    return {
        "full_mean_coverage": full["coverage"],
        "full_mean_mae": full["mae"],
        "full_mean_rmse": full["rmse"],
        "full_mean_abs_rel": full["abs_rel"],
        "full_mean_delta_1.10": full["delta_1.10"],
        "full_mean_delta_1.25": full["delta_1.25"],
        "overlap_mean_mae": overlap["mae"],
        "overlap_mean_rmse": overlap["rmse"],
        "overlap_mean_abs_rel": overlap["abs_rel"],
        "overlap_mean_delta_1.10": overlap["delta_1.10"],
        "overlap_mean_delta_1.25": overlap["delta_1.25"],
    }


def add_scores(
    metrics: dict[str, float],
    baseline_full: dict[str, float],
    baseline_overlap: dict[str, float],
) -> dict[str, float]:
    row = dict(metrics)
    row["coverage_gain_pct"] = (
        (metrics["full_mean_coverage"] - baseline_full["coverage"]) / baseline_full["coverage"] * 100.0
    )
    row["overlap_rmse_improvement_pct"] = (
        (baseline_overlap["rmse"] - metrics["overlap_mean_rmse"]) / baseline_overlap["rmse"] * 100.0
    )
    row["overlap_mae_improvement_pct"] = (
        (baseline_overlap["mae"] - metrics["overlap_mean_mae"]) / baseline_overlap["mae"] * 100.0
    )
    row["full_rmse_degradation_pct"] = (
        (metrics["full_mean_rmse"] - baseline_full["rmse"]) / baseline_full["rmse"] * 100.0
    )
    row["full_mae_degradation_pct"] = (
        (metrics["full_mean_mae"] - baseline_full["mae"]) / baseline_full["mae"] * 100.0
    )
    row["tradeoff_score"] = (
        row["coverage_gain_pct"]
        - 2.0 * max(0.0, row["full_rmse_degradation_pct"])
        - 1.0 * max(0.0, row["full_mae_degradation_pct"])
    )
    return row


def load_frames(
    stems: list[str],
    wmf_module,
    rgb_dir: Path,
    tof_dir: Path,
    amplitude_dir: Path,
    gt_dir: Path,
    baseline_dir: Path,
) -> list[dict[str, np.ndarray]]:
    frames: list[dict[str, np.ndarray]] = []
    for stem in stems:
        rgb = wmf_module.read_rgb(rgb_dir / f"{stem}.png")
        tof = wmf_module.read_u16(tof_dir / f"{stem}.png")
        amplitude = wmf_module.read_u16(amplitude_dir / f"{stem}.png")
        gt = wmf_module.read_u16(gt_dir / f"{stem}.png")
        baseline = wmf_module.read_u16(baseline_dir / f"{stem}.png")
        rgb_small = cv2.resize(rgb, (tof.shape[1], tof.shape[0]), interpolation=cv2.INTER_AREA)
        frames.append(
            {
                "stem": stem,
                "rgb_small": rgb_small,
                "tof": tof,
                "amplitude": amplitude,
                "gt": gt,
                "baseline": baseline,
            }
        )
    return frames


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rgb_dir = root / args.rgb_dir
    tof_dir = root / args.tof_dir
    amplitude_dir = root / args.amplitude_dir
    gt_dir = root / args.gt_dir
    baseline_dir = root / args.baseline_dir

    stems = common_stems(rgb_dir, tof_dir, amplitude_dir, gt_dir, baseline_dir)
    frames = load_frames(stems, wmf_module, rgb_dir, tof_dir, amplitude_dir, gt_dir, baseline_dir)

    baseline_full = {
        "coverage": 0.4554038065843622,
        "mae": 94.45802552023052,
        "rmse": 107.50608646722488,
    }
    baseline_overlap = {
        "mae": 94.45802552023052,
        "rmse": 107.50608646722488,
    }

    grid: list[dict[str, float]] = []
    for radius in [1, 2, 3]:
        for sigma_spatial in [1.5, 2.0, 2.5]:
            for sigma_color in [12.0, 16.0, 20.0, 24.0, 30.0]:
                grid.append(
                    {
                        "radius": float(radius),
                        "sigma_spatial": sigma_spatial,
                        "sigma_color": sigma_color,
                        "min_amplitude": 1.0,
                    }
                )

    results = []
    for index, params in enumerate(grid, start=1):
        metrics = evaluate_config(frames, params, wmf_module)
        scored = add_scores(metrics, baseline_full, baseline_overlap)
        scored["params"] = params
        results.append(scored)
        print(
            f"[{index:02d}/{len(grid):02d}] "
            f"r={int(params['radius'])} ss={params['sigma_spatial']} sc={params['sigma_color']} "
            f"overlap_rmse_imp={scored['overlap_rmse_improvement_pct']:.2f}% "
            f"coverage_gain={scored['coverage_gain_pct']:.2f}% "
            f"full_rmse_deg={scored['full_rmse_degradation_pct']:.2f}%"
        )

    best_overlap = max(
        results,
        key=lambda row: (
            row["overlap_rmse_improvement_pct"],
            row["overlap_mae_improvement_pct"],
        ),
    )
    best_tradeoff = max(results, key=lambda row: row["tradeoff_score"])

    payload = {
        "best_overlap": best_overlap,
        "best_tradeoff": best_tradeoff,
        "results": sorted(results, key=lambda row: (-row["tradeoff_score"], -row["overlap_rmse_improvement_pct"])),
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved search results to {args.output_json.resolve()}")
    print("Best overlap params:", best_overlap["params"])
    print("Best tradeoff params:", best_tradeoff["params"])


if __name__ == "__main__":
    main()
