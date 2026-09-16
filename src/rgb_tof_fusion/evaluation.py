from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fused depth maps against ground-truth depth maps."
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory with predicted depth PNGs.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("depthmaps"),
        help="Directory with ground-truth depth PNGs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the benchmark summary as JSON.",
    )
    parser.add_argument(
        "--pred-valid-min",
        type=int,
        default=1,
        help="Predicted pixels below this value are treated as invalid.",
    )
    parser.add_argument(
        "--gt-valid-min",
        type=int,
        default=1,
        help="GT pixels below this value are treated as invalid.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor applied to prediction before evaluation.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Optional reference prediction directory for overlap/filled-region evaluation.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["pred-valid", "overlap-only", "filled-region-only"],
        default="pred-valid",
        help="How to choose evaluation pixels when --reference-dir is provided.",
    )
    return parser.parse_args()


def read_u16(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if image.dtype != np.uint16:
        raise ValueError(f"Expected uint16 PNG, got {image.dtype} for {path}")
    return image


def list_common_stems(pred_dir: Path, gt_dir: Path) -> list[str]:
    pred_stems = {path.stem for path in pred_dir.glob("*.png")}
    gt_stems = {path.stem for path in gt_dir.glob("*.png")}
    return sorted(pred_stems & gt_stems, key=lambda name: int(name))


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)

    diff = pred_f - gt_f
    abs_diff = np.abs(diff)
    sq_diff = diff * diff

    mae = float(abs_diff.mean())
    rmse = float(np.sqrt(sq_diff.mean()))
    abs_rel = float((abs_diff / np.clip(gt_f, 1e-6, None)).mean())

    pred_safe = np.clip(pred_f, 1e-6, None)
    gt_safe = np.clip(gt_f, 1e-6, None)
    ratio = np.maximum(pred_safe / gt_safe, gt_safe / pred_safe)
    delta_1_05 = float((ratio < 1.05).mean())
    delta_1_10 = float((ratio < 1.10).mean())
    delta_1_25 = float((ratio < 1.25).mean())

    return {
        "mae": mae,
        "rmse": rmse,
        "abs_rel": abs_rel,
        "delta_1.05": delta_1_05,
        "delta_1.10": delta_1_10,
        "delta_1.25": delta_1_25,
    }


def main() -> None:
    args = parse_args()
    pred_dir = args.pred_dir.resolve()
    gt_dir = args.gt_dir.resolve()

    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
    reference_dir = args.reference_dir.resolve() if args.reference_dir is not None else None
    if reference_dir is not None and not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

    stems = list_common_stems(pred_dir, gt_dir)
    if reference_dir is not None:
        stems = [stem for stem in stems if (reference_dir / f"{stem}.png").is_file()]
    if not stems:
        raise RuntimeError("No common PNG filenames found between prediction and GT directories.")

    per_frame: dict[str, dict[str, float]] = {}
    metrics_accumulator: dict[str, list[float]] = {
        "mae": [],
        "rmse": [],
        "abs_rel": [],
        "delta_1.05": [],
        "delta_1.10": [],
        "delta_1.25": [],
        "coverage": [],
        "valid_pixels": [],
    }

    for stem in stems:
        pred_raw = read_u16(pred_dir / f"{stem}.png")
        gt_raw = read_u16(gt_dir / f"{stem}.png")

        if pred_raw.shape != gt_raw.shape:
            raise ValueError(
                f"Shape mismatch for frame {stem}: pred {pred_raw.shape}, gt {gt_raw.shape}"
            )

        pred_scaled = pred_raw.astype(np.float32) * args.scale
        pred = np.rint(pred_scaled).astype(np.uint16)

        pred_valid = pred >= args.pred_valid_min
        gt_valid = gt_raw >= args.gt_valid_min
        valid = pred_valid & gt_valid

        if reference_dir is not None:
            ref_raw = read_u16(reference_dir / f"{stem}.png")
            if ref_raw.shape != gt_raw.shape:
                raise ValueError(
                    f"Shape mismatch for reference frame {stem}: ref {ref_raw.shape}, gt {gt_raw.shape}"
                )
            ref_valid = ref_raw >= args.pred_valid_min
            if args.mask_mode == "overlap-only":
                valid = valid & ref_valid
            elif args.mask_mode == "filled-region-only":
                valid = valid & (~ref_valid)

        coverage = float(valid.mean())
        valid_pixels = int(valid.sum())
        metrics_accumulator["coverage"].append(coverage)
        metrics_accumulator["valid_pixels"].append(float(valid_pixels))

        if valid_pixels == 0:
            per_frame[stem] = {
                "coverage": coverage,
                "valid_pixels": valid_pixels,
            }
            continue

        frame_metrics = compute_metrics(pred[valid], gt_raw[valid])
        frame_metrics["coverage"] = coverage
        frame_metrics["valid_pixels"] = float(valid_pixels)
        per_frame[stem] = frame_metrics

        for key in ["mae", "rmse", "abs_rel", "delta_1.05", "delta_1.10", "delta_1.25"]:
            metrics_accumulator[key].append(frame_metrics[key])

    summary = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "reference_dir": str(reference_dir) if reference_dir is not None else None,
        "mask_mode": args.mask_mode,
        "num_common_frames": len(stems),
        "num_scored_frames": len(metrics_accumulator["mae"]),
        "mean_coverage": float(np.mean(metrics_accumulator["coverage"])),
        "mean_valid_pixels": float(np.mean(metrics_accumulator["valid_pixels"])),
    }

    for key in ["mae", "rmse", "abs_rel", "delta_1.05", "delta_1.10", "delta_1.25"]:
        values = metrics_accumulator[key]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
        summary[f"median_{key}"] = float(np.median(values)) if values else None

    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        payload = {
            "summary": summary,
            "per_frame": per_frame,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
