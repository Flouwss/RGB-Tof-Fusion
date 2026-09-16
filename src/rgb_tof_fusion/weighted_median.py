from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RGB-guided weighted median filter for ToF frames."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Dataset root.")
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--tof-dir", type=str, default="tof_rect")
    parser.add_argument("--amplitude-dir", type=str, default="amplitude_rect")
    parser.add_argument("--output-dir", type=str, default="tof_wmf_rgb_guided")
    parser.add_argument(
        "--upsampled-output-dir",
        type=str,
        default="tof_wmf_rgb_guided_upsampled",
        help="Set to empty string to skip upsampled output.",
    )
    parser.add_argument("--radius", type=int, default=3, help="Window radius.")
    parser.add_argument(
        "--sigma-spatial",
        type=float,
        default=2.0,
        help="Spatial Gaussian sigma in ToF pixels.",
    )
    parser.add_argument(
        "--sigma-color",
        type=float,
        default=12.0,
        help="RGB Gaussian sigma in 0..255 color space.",
    )
    parser.add_argument(
        "--min-amplitude",
        type=int,
        default=1,
        help="Pixels below this amplitude are treated as invalid.",
    )
    parser.add_argument(
        "--upsample-mode",
        choices=["nearest", "linear"],
        default="nearest",
        help="Interpolation for optional upsampled output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N frames, 0 means all.",
    )
    return parser.parse_args()


def list_common_stems(*dirs: Path) -> list[str]:
    stems = []
    for directory in dirs:
        stems.append({path.stem for path in directory.glob("*.png")})
    return sorted(set.intersection(*stems), key=lambda name: int(name))


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


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=-1)
    sorted_values = np.take_along_axis(values, order, axis=-1)
    sorted_weights = np.take_along_axis(weights, order, axis=-1)
    cumulative = np.cumsum(sorted_weights, axis=-1)
    half = cumulative[..., -1] * 0.5
    index = (cumulative >= half[..., None]).argmax(axis=-1)
    result = np.take_along_axis(sorted_values, index[..., None], axis=-1)[..., 0]
    result[cumulative[..., -1] <= 0] = 0
    return result


def joint_weighted_median_filter(
    tof: np.ndarray,
    rgb_guidance: np.ndarray,
    valid_mask: np.ndarray,
    radius: int,
    sigma_spatial: float,
    sigma_color: float,
) -> np.ndarray:
    if tof.shape != valid_mask.shape:
        raise ValueError("ToF and valid mask shapes must match.")
    if rgb_guidance.shape[:2] != tof.shape:
        raise ValueError("RGB guidance must match ToF resolution.")

    h, w = tof.shape
    pad = radius

    tof_pad = np.pad(tof.astype(np.float32), pad, mode="edge")
    mask_pad = np.pad(valid_mask.astype(np.float32), pad, mode="constant")
    rgb_pad = np.pad(rgb_guidance.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="edge")

    center_rgb = rgb_guidance.astype(np.float32)
    offsets: list[tuple[int, int]] = []
    spatial_weights: list[float] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offsets.append((dy, dx))
            dist2 = float(dx * dx + dy * dy)
            spatial_weights.append(np.exp(-dist2 / (2.0 * sigma_spatial * sigma_spatial)))

    num_offsets = len(offsets)
    value_stack = np.empty((h, w, num_offsets), dtype=np.float32)
    weight_stack = np.empty((h, w, num_offsets), dtype=np.float32)

    color_scale = 2.0 * sigma_color * sigma_color
    for idx, ((dy, dx), spatial_weight) in enumerate(zip(offsets, spatial_weights)):
        y0 = pad + dy
        x0 = pad + dx
        shifted_tof = tof_pad[y0 : y0 + h, x0 : x0 + w]
        shifted_mask = mask_pad[y0 : y0 + h, x0 : x0 + w]
        shifted_rgb = rgb_pad[y0 : y0 + h, x0 : x0 + w]

        color_diff2 = np.sum((shifted_rgb - center_rgb) ** 2, axis=-1)
        color_weight = np.exp(-color_diff2 / color_scale)
        weights = spatial_weight * color_weight * shifted_mask

        value_stack[..., idx] = shifted_tof
        weight_stack[..., idx] = weights

    filtered = weighted_median(value_stack, weight_stack)
    return np.rint(filtered).astype(np.uint16)


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
    upsampled_output_dir = root / args.upsampled_output_dir if args.upsampled_output_dir else None

    for directory in [rgb_dir, tof_dir, amplitude_dir]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing directory: {directory}")

    ensure_dir(output_dir)
    if upsampled_output_dir is not None:
        ensure_dir(upsampled_output_dir)
    save_run_config(
        args,
        [output_dir] + ([upsampled_output_dir] if upsampled_output_dir is not None else []),
    )

    stems = list_common_stems(rgb_dir, tof_dir, amplitude_dir)
    if args.limit > 0:
        stems = stems[: args.limit]
    if not stems:
        raise RuntimeError("No matching RGB/ToF/Amplitude PNG triplets found.")

    interpolation = cv2.INTER_NEAREST if args.upsample_mode == "nearest" else cv2.INTER_LINEAR

    print(f"Processing {len(stems)} frames from {root}")
    print(
        f"radius={args.radius}, sigma_spatial={args.sigma_spatial}, "
        f"sigma_color={args.sigma_color}, min_amplitude={args.min_amplitude}"
    )

    for idx, stem in enumerate(stems, start=1):
        rgb = read_rgb(rgb_dir / f"{stem}.png")
        tof = read_u16(tof_dir / f"{stem}.png")
        amplitude = read_u16(amplitude_dir / f"{stem}.png")

        rgb_small = cv2.resize(rgb, (tof.shape[1], tof.shape[0]), interpolation=cv2.INTER_AREA)
        valid_mask = (tof > 0) & (amplitude >= args.min_amplitude)

        filtered = joint_weighted_median_filter(
            tof=tof,
            rgb_guidance=rgb_small,
            valid_mask=valid_mask,
            radius=args.radius,
            sigma_spatial=args.sigma_spatial,
            sigma_color=args.sigma_color,
        )

        cv2.imwrite(str(output_dir / f"{stem}.png"), filtered)

        if upsampled_output_dir is not None:
            upsampled = cv2.resize(filtered, (rgb.shape[1], rgb.shape[0]), interpolation=interpolation)
            cv2.imwrite(str(upsampled_output_dir / f"{stem}.png"), upsampled)

        print(f"[{idx:03d}/{len(stems):03d}] {stem}")

    print("Done.")


if __name__ == "__main__":
    main()
