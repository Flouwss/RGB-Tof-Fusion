from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .depth_completion import (
    confidence_aware_fill,
    normalize_confidence,
    read_rgb,
    read_u16,
)
from .weighted_median import joint_weighted_median_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage pipeline: confidence-aware completion followed by RGB-guided WMF."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--tof-dir", type=str, default="tof_rect")
    parser.add_argument("--amplitude-dir", type=str, default="amplitude_rect")
    parser.add_argument("--output-dir", type=str, default="tof_confidence_wmf_pipeline")
    parser.add_argument(
        "--upsampled-output-dir",
        type=str,
        default="tof_confidence_wmf_pipeline_upsampled",
    )
    parser.add_argument("--min-amplitude", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--cc-confidence-percentile", type=float, default=95.0)
    parser.add_argument("--cc-confidence-gamma", type=float, default=0.7)
    parser.add_argument("--cc-radius", type=int, default=1)
    parser.add_argument("--cc-sigma-spatial", type=float, default=1.0)
    parser.add_argument("--cc-sigma-color", type=float, default=30.0)
    parser.add_argument("--cc-fill-confidence-decay", type=float, default=0.6)
    parser.add_argument("--cc-min-normalized-weight", type=float, default=0.1)
    parser.add_argument("--cc-max-iters", type=int, default=4)
    parser.add_argument("--cc-refine", action="store_true")

    parser.add_argument("--wmf-radius", type=int, default=2)
    parser.add_argument("--wmf-sigma-spatial", type=float, default=2.5)
    parser.add_argument("--wmf-sigma-color", type=float, default=30.0)
    parser.add_argument(
        "--upsample-mode",
        choices=["nearest", "linear"],
        default="nearest",
    )
    return parser.parse_args()


def list_common_stems(*dirs: Path) -> list[str]:
    sets = [{path.stem for path in directory.glob("*.png")} for directory in dirs]
    return sorted(set.intersection(*sets), key=lambda name: int(name))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_run_config(args: argparse.Namespace, target_dirs: list[Path]) -> None:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    for directory in target_dirs:
        (directory / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rgb_dir = root / args.rgb_dir
    tof_dir = root / args.tof_dir
    amplitude_dir = root / args.amplitude_dir
    output_dir = root / args.output_dir
    upsampled_output_dir = root / args.upsampled_output_dir

    for directory in [rgb_dir, tof_dir, amplitude_dir]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing directory: {directory}")

    ensure_dir(output_dir)
    ensure_dir(upsampled_output_dir)
    save_run_config(args, [output_dir, upsampled_output_dir])

    stems = list_common_stems(rgb_dir, tof_dir, amplitude_dir)
    if args.limit > 0:
        stems = stems[: args.limit]
    if not stems:
        raise RuntimeError("No matching RGB/ToF/Amplitude PNG triplets found.")

    interpolation = cv2.INTER_NEAREST if args.upsample_mode == "nearest" else cv2.INTER_LINEAR

    print(f"Processing {len(stems)} frames from {root}")
    print(
        "completion: "
        f"radius={args.cc_radius}, sigma_spatial={args.cc_sigma_spatial}, sigma_color={args.cc_sigma_color}, "
        f"decay={args.cc_fill_confidence_decay}, min_weight={args.cc_min_normalized_weight}, "
        f"max_iters={args.cc_max_iters}, refine={args.cc_refine}"
    )
    print(
        "wmf: "
        f"radius={args.wmf_radius}, sigma_spatial={args.wmf_sigma_spatial}, sigma_color={args.wmf_sigma_color}"
    )

    for index, stem in enumerate(stems, start=1):
        rgb = read_rgb(rgb_dir / f"{stem}.png")
        tof = read_u16(tof_dir / f"{stem}.png")
        amplitude = read_u16(amplitude_dir / f"{stem}.png")

        rgb_small = cv2.resize(rgb, (tof.shape[1], tof.shape[0]), interpolation=cv2.INTER_AREA)
        valid_mask = (tof > 0) & (amplitude >= args.min_amplitude)
        confidence = normalize_confidence(
            amplitude=amplitude,
            valid_mask=valid_mask,
            percentile=args.cc_confidence_percentile,
            gamma=args.cc_confidence_gamma,
        )

        completed_small = confidence_aware_fill(
            depth=tof,
            rgb=rgb_small,
            confidence=confidence,
            valid_mask=valid_mask,
            radius=args.cc_radius,
            sigma_spatial=args.cc_sigma_spatial,
            sigma_color=args.cc_sigma_color,
            max_iters=args.cc_max_iters,
            fill_confidence_decay=args.cc_fill_confidence_decay,
            min_normalized_weight=args.cc_min_normalized_weight,
            refine=args.cc_refine,
        )

        refined_small = joint_weighted_median_filter(
            tof=completed_small,
            rgb_guidance=rgb_small,
            valid_mask=(completed_small > 0),
            radius=args.wmf_radius,
            sigma_spatial=args.wmf_sigma_spatial,
            sigma_color=args.wmf_sigma_color,
        )

        refined_up = cv2.resize(
            refined_small,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=interpolation,
        )

        cv2.imwrite(str(output_dir / f"{stem}.png"), refined_small)
        cv2.imwrite(str(upsampled_output_dir / f"{stem}.png"), refined_up)
        print(f"[{index:03d}/{len(stems):03d}] {stem}")

    print("Done.")


if __name__ == "__main__":
    main()
