from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confidence/amplitude-aware depth completion guided by RGB."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--tof-dir", type=str, default="tof_rect")
    parser.add_argument("--amplitude-dir", type=str, default="amplitude_rect")
    parser.add_argument("--output-dir", type=str, default="tof_confidence_completion")
    parser.add_argument(
        "--upsampled-output-dir",
        type=str,
        default="tof_confidence_completion_upsampled",
    )
    parser.add_argument("--min-amplitude", type=int, default=1)
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=95.0,
        help="Upper percentile for amplitude normalization.",
    )
    parser.add_argument("--confidence-gamma", type=float, default=0.7)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--sigma-spatial", type=float, default=2.0)
    parser.add_argument("--sigma-color", type=float, default=20.0)
    parser.add_argument(
        "--fill-confidence-decay",
        type=float,
        default=0.9,
        help="Confidence assigned to newly filled pixels.",
    )
    parser.add_argument(
        "--min-normalized-weight",
        type=float,
        default=0.02,
        help="Minimum normalized support to accept a fill.",
    )
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Apply one confidence-aware smoothing pass after filling.",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def list_common_stems(*dirs: Path) -> list[str]:
    sets = [{path.stem for path in directory.glob("*.png")} for directory in dirs]
    return sorted(set.intersection(*sets), key=lambda name: int(name))


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_u16(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read 16-bit image: {path}")
    if image.dtype != np.uint16:
        raise ValueError(f"Expected uint16 PNG, got {image.dtype} for {path}")
    return image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_run_config(args: argparse.Namespace, target_dirs: list[Path]) -> None:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    for directory in target_dirs:
        (directory / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def normalize_confidence(amplitude: np.ndarray, valid_mask: np.ndarray, percentile: float, gamma: float) -> np.ndarray:
    confidence = np.zeros_like(amplitude, dtype=np.float32)
    if not np.any(valid_mask):
        return confidence
    valid_values = amplitude[valid_mask].astype(np.float32)
    scale = np.percentile(valid_values, percentile)
    scale = max(float(scale), 1.0)
    confidence[valid_mask] = np.clip(valid_values / scale, 0.0, 1.0)
    if gamma != 1.0:
        confidence[valid_mask] = np.power(confidence[valid_mask], gamma)
    return confidence


def build_offsets(radius: int, sigma_spatial: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    offsets: list[tuple[int, int]] = []
    spatial_weights = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offsets.append((dy, dx))
            dist2 = float(dx * dx + dy * dy)
            spatial_weights.append(np.exp(-dist2 / (2.0 * sigma_spatial * sigma_spatial)))
    return offsets, np.asarray(spatial_weights, dtype=np.float32)


def confidence_aware_fill(
    depth: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    valid_mask: np.ndarray,
    radius: int,
    sigma_spatial: float,
    sigma_color: float,
    max_iters: int,
    fill_confidence_decay: float,
    min_normalized_weight: float,
    refine: bool,
) -> np.ndarray:
    filled = depth.astype(np.float32).copy()
    conf = confidence.astype(np.float32).copy()
    valid = valid_mask.copy()

    offsets, spatial_weights = build_offsets(radius, sigma_spatial)
    pad = radius
    rgb_f = rgb.astype(np.float32)
    color_scale = 2.0 * sigma_color * sigma_color

    for _ in range(max_iters):
        pending = ~valid
        if not np.any(pending):
            break

        value_num = np.zeros_like(filled, dtype=np.float32)
        weight_sum = np.zeros_like(filled, dtype=np.float32)
        support_sum = np.zeros_like(filled, dtype=np.float32)

        filled_pad = np.pad(filled, pad, mode="edge")
        conf_pad = np.pad(conf, pad, mode="constant")
        valid_pad = np.pad(valid.astype(np.float32), pad, mode="constant")
        rgb_pad = np.pad(rgb_f, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

        for (dy, dx), spatial_weight in zip(offsets, spatial_weights):
            y0 = pad + dy
            x0 = pad + dx

            shifted_depth = filled_pad[y0 : y0 + filled.shape[0], x0 : x0 + filled.shape[1]]
            shifted_conf = conf_pad[y0 : y0 + filled.shape[0], x0 : x0 + filled.shape[1]]
            shifted_valid = valid_pad[y0 : y0 + filled.shape[0], x0 : x0 + filled.shape[1]]
            shifted_rgb = rgb_pad[y0 : y0 + filled.shape[0], x0 : x0 + filled.shape[1]]

            color_diff2 = np.sum((shifted_rgb - rgb_f) ** 2, axis=-1)
            color_weight = np.exp(-color_diff2 / color_scale)
            weight = spatial_weight * color_weight * shifted_conf * shifted_valid

            value_num += weight * shifted_depth
            weight_sum += weight
            support_sum += spatial_weight * shifted_valid

        normalized_support = np.divide(
            weight_sum,
            np.maximum(support_sum, 1e-6),
            out=np.zeros_like(weight_sum),
            where=support_sum > 0,
        )
        accepted = pending & (normalized_support >= min_normalized_weight) & (weight_sum > 0)
        if not np.any(accepted):
            break

        filled[accepted] = value_num[accepted] / weight_sum[accepted]
        conf[accepted] = np.clip(normalized_support[accepted] * fill_confidence_decay, 0.0, 1.0)
        valid[accepted] = True

    if refine:
        value_num = np.zeros_like(filled, dtype=np.float32)
        weight_sum = np.zeros_like(filled, dtype=np.float32)
        filled_pad = np.pad(filled, pad, mode="edge")
        conf_pad = np.pad(conf, pad, mode="constant")
        valid_pad = np.pad(valid.astype(np.float32), pad, mode="constant")
        rgb_pad = np.pad(rgb_f, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

        for (dy, dx), spatial_weight in zip(offsets, spatial_weights):
            y0 = pad + dy
            x0 = pad + dx
            shifted_depth = filled_pad[y0 : y0 + filled.shape[0], x0 : x0 + filled.shape[1]]
            shifted_conf = conf_pad[y0 : y0 + filled.shape[0], x0 : x0 + filled.shape[1]]
            shifted_valid = valid_pad[y0 : y0 + filled.shape[0], x0 : x0 + filled.shape[1]]
            shifted_rgb = rgb_pad[y0 : y0 + filled.shape[0], x0 : x0 + filled.shape[1]]
            color_diff2 = np.sum((shifted_rgb - rgb_f) ** 2, axis=-1)
            color_weight = np.exp(-color_diff2 / color_scale)
            weight = spatial_weight * color_weight * shifted_conf * shifted_valid
            value_num += weight * shifted_depth
            weight_sum += weight

        smoothed = np.divide(
            value_num,
            np.maximum(weight_sum, 1e-6),
            out=filled.copy(),
            where=weight_sum > 0,
        )
        filled[valid] = smoothed[valid]

    return np.rint(filled).astype(np.uint16)


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

    print(f"Processing {len(stems)} frames from {root}")
    print(
        f"radius={args.radius}, sigma_spatial={args.sigma_spatial}, sigma_color={args.sigma_color}, "
        f"min_amplitude={args.min_amplitude}, max_iters={args.max_iters}, refine={args.refine}"
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
            percentile=args.confidence_percentile,
            gamma=args.confidence_gamma,
        )

        completed_small = confidence_aware_fill(
            depth=tof,
            rgb=rgb_small,
            confidence=confidence,
            valid_mask=valid_mask,
            radius=args.radius,
            sigma_spatial=args.sigma_spatial,
            sigma_color=args.sigma_color,
            max_iters=args.max_iters,
            fill_confidence_decay=args.fill_confidence_decay,
            min_normalized_weight=args.min_normalized_weight,
            refine=args.refine,
        )

        completed_up = cv2.resize(
            completed_small,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.imwrite(str(output_dir / f"{stem}.png"), completed_small)
        cv2.imwrite(str(upsampled_output_dir / f"{stem}.png"), completed_up)
        print(f"[{index:03d}/{len(stems):03d}] {stem}")

    print("Done.")


if __name__ == "__main__":
    main()
