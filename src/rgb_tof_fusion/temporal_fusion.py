from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .depth_completion import normalize_confidence, read_rgb, read_u16
from .weighted_median import weighted_median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal multi-frame fusion for aligned ToF frames."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--tof-dir", type=str, default="tof_rect")
    parser.add_argument("--amplitude-dir", type=str, default="amplitude_rect")
    parser.add_argument("--output-dir", type=str, default="tof_multiframe_fusion")
    parser.add_argument(
        "--upsampled-output-dir",
        type=str,
        default="tof_multiframe_fusion_upsampled",
    )
    parser.add_argument("--min-amplitude", type=int, default=1)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--sigma-temporal", type=float, default=1.0)
    parser.add_argument("--sigma-color", type=float, default=18.0)
    parser.add_argument("--confidence-percentile", type=float, default=95.0)
    parser.add_argument("--confidence-gamma", type=float, default=0.7)
    parser.add_argument(
        "--fuse-mode",
        choices=["weighted-median", "weighted-average"],
        default="weighted-median",
    )
    parser.add_argument(
        "--preserve-center-valid",
        action="store_true",
        help="Keep original center-frame valid pixels unchanged and fill only holes.",
    )
    parser.add_argument(
        "--upsample-mode",
        choices=["nearest", "linear"],
        default="nearest",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_run_config(args: argparse.Namespace, target_dirs: list[Path]) -> None:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    for directory in target_dirs:
        (directory / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def list_common_stems(*dirs: Path) -> list[str]:
    sets = [{path.stem for path in directory.glob("*.png")} for directory in dirs]
    return sorted(set.intersection(*sets), key=lambda name: int(name))


def weighted_average(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight_sum = weights.sum(axis=-1)
    weighted_sum = (values * weights).sum(axis=-1)
    result = np.divide(
        weighted_sum,
        np.maximum(weight_sum, 1e-6),
        out=np.zeros_like(weighted_sum),
        where=weight_sum > 0,
    )
    return result.astype(np.float32)


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

    frames = []
    for stem in stems:
        rgb = read_rgb(rgb_dir / f"{stem}.png")
        tof = read_u16(tof_dir / f"{stem}.png")
        amplitude = read_u16(amplitude_dir / f"{stem}.png")
        rgb_small = cv2.resize(rgb, (tof.shape[1], tof.shape[0]), interpolation=cv2.INTER_AREA)
        valid_mask = (tof > 0) & (amplitude >= args.min_amplitude)
        confidence = normalize_confidence(
            amplitude=amplitude,
            valid_mask=valid_mask,
            percentile=args.confidence_percentile,
            gamma=args.confidence_gamma,
        )
        frames.append(
            {
                "stem": stem,
                "rgb_full": rgb,
                "rgb_small": rgb_small.astype(np.float32),
                "tof": tof.astype(np.float32),
                "valid_mask": valid_mask,
                "confidence": confidence.astype(np.float32),
            }
        )

    interpolation = cv2.INTER_NEAREST if args.upsample_mode == "nearest" else cv2.INTER_LINEAR
    color_scale = 2.0 * args.sigma_color * args.sigma_color

    print(f"Processing {len(frames)} frames from {root}")
    print(
        f"temporal_radius={args.temporal_radius}, sigma_temporal={args.sigma_temporal}, "
        f"sigma_color={args.sigma_color}, fuse_mode={args.fuse_mode}, "
        f"preserve_center_valid={args.preserve_center_valid}"
    )

    for center_idx, center in enumerate(frames):
        start = max(0, center_idx - args.temporal_radius)
        end = min(len(frames), center_idx + args.temporal_radius + 1)
        neighbors = frames[start:end]

        value_stack = []
        weight_stack = []

        for neighbor_idx, neighbor in enumerate(neighbors, start=start):
            dt = abs(neighbor_idx - center_idx)
            temporal_weight = np.exp(-(dt * dt) / (2.0 * args.sigma_temporal * args.sigma_temporal))
            color_diff2 = np.sum((neighbor["rgb_small"] - center["rgb_small"]) ** 2, axis=-1)
            color_weight = np.exp(-color_diff2 / color_scale)
            weights = temporal_weight * color_weight * neighbor["confidence"] * neighbor["valid_mask"].astype(np.float32)

            value_stack.append(neighbor["tof"])
            weight_stack.append(weights)

        values = np.stack(value_stack, axis=-1)
        weights = np.stack(weight_stack, axis=-1)

        if args.fuse_mode == "weighted-average":
            fused = weighted_average(values, weights)
        else:
            fused = weighted_median(values, weights)

        fused = np.rint(fused).astype(np.uint16)
        if args.preserve_center_valid:
            fused[center["valid_mask"]] = center["tof"][center["valid_mask"]].astype(np.uint16)

        fused_up = cv2.resize(
            fused,
            (center["rgb_full"].shape[1], center["rgb_full"].shape[0]),
            interpolation=interpolation,
        )

        cv2.imwrite(str(output_dir / f"{center['stem']}.png"), fused)
        cv2.imwrite(str(upsampled_output_dir / f"{center['stem']}.png"), fused_up)
        print(f"[{center_idx + 1:03d}/{len(frames):03d}] {center['stem']}")

    print("Done.")


if __name__ == "__main__":
    main()
