from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .depth_completion import read_rgb, read_u16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Superpixel-guided plane fitting for ToF completion."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--tof-dir", type=str, default="tof_rect")
    parser.add_argument("--amplitude-dir", type=str, default="amplitude_rect")
    parser.add_argument("--output-dir", type=str, default="tof_superpixel_plane")
    parser.add_argument(
        "--upsampled-output-dir",
        type=str,
        default="tof_superpixel_plane_upsampled",
    )
    parser.add_argument("--min-amplitude", type=int, default=1)
    parser.add_argument("--num-superpixels", type=int, default=45)
    parser.add_argument("--compactness", type=float, default=8.0)
    parser.add_argument("--plane-min-points", type=int, default=10)
    parser.add_argument("--plane-max-residual", type=float, default=22.0)
    parser.add_argument("--plane-max-gradient", type=float, default=80.0)
    parser.add_argument("--plane-blend-valid", type=float, default=0.2)
    parser.add_argument("--plane-fill-missing-only", action="store_true")
    parser.add_argument("--post-median-ksize", type=int, default=3)
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


def compute_superpixels(rgb_small: np.ndarray, num_superpixels: int, compactness: float) -> np.ndarray:
    h, w = rgb_small.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    rgb_feat = rgb_small.astype(np.float32).reshape(-1, 3) / 255.0
    xy_feat = np.stack([xx / max(w - 1, 1), yy / max(h - 1, 1)], axis=-1).reshape(-1, 2)
    features = np.concatenate([rgb_feat, compactness * xy_feat], axis=1).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, _ = cv2.kmeans(
        features,
        num_superpixels,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.reshape(h, w)


def fit_plane(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> tuple[np.ndarray, float, float]:
    A = np.stack([xx, yy, np.ones_like(xx)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(A, zz, rcond=None)
    pred = A @ coeffs
    residual = float(np.median(np.abs(pred - zz)))
    gradient = float(np.sqrt(coeffs[0] * coeffs[0] + coeffs[1] * coeffs[1]))
    return coeffs, residual, gradient


def superpixel_plane_complete(
    depth: np.ndarray,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    plane_min_points: int,
    plane_max_residual: float,
    plane_max_gradient: float,
    plane_blend_valid: float,
    plane_fill_missing_only: bool,
) -> np.ndarray:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    depth_f = depth.astype(np.float32)
    output = depth_f.copy()

    for label in np.unique(labels):
        seg = labels == label
        seg_valid = seg & valid_mask
        if int(seg_valid.sum()) < plane_min_points:
            continue

        coeffs, residual, gradient = fit_plane(
            xx[seg_valid].astype(np.float32),
            yy[seg_valid].astype(np.float32),
            depth_f[seg_valid].astype(np.float32),
        )
        if residual > plane_max_residual or gradient > plane_max_gradient:
            continue

        plane_values = coeffs[0] * xx[seg] + coeffs[1] * yy[seg] + coeffs[2]
        plane_values = np.clip(plane_values, 0, np.iinfo(np.uint16).max)
        seg_values = output[seg]
        seg_valid_mask = valid_mask[seg]

        seg_values[~seg_valid_mask] = plane_values[~seg_valid_mask]

        if not plane_fill_missing_only:
            seg_values[seg_valid_mask] = (
                (1.0 - plane_blend_valid) * depth_f[seg][seg_valid_mask]
                + plane_blend_valid * plane_values[seg_valid_mask]
            )

        output[seg] = seg_values

    return np.rint(output).astype(np.uint16)


def postprocess(depth: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return depth
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(depth, ksize)


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
        f"superpixels={args.num_superpixels}, compactness={args.compactness}, "
        f"plane_min_points={args.plane_min_points}, plane_max_residual={args.plane_max_residual}, "
        f"plane_max_gradient={args.plane_max_gradient}, fill_missing_only={args.plane_fill_missing_only}"
    )

    for index, stem in enumerate(stems, start=1):
        rgb = read_rgb(rgb_dir / f"{stem}.png")
        tof = read_u16(tof_dir / f"{stem}.png")
        amplitude = read_u16(amplitude_dir / f"{stem}.png")

        rgb_small = cv2.resize(rgb, (tof.shape[1], tof.shape[0]), interpolation=cv2.INTER_AREA)
        valid_mask = (tof > 0) & (amplitude >= args.min_amplitude)
        labels = compute_superpixels(
            rgb_small=rgb_small,
            num_superpixels=args.num_superpixels,
            compactness=args.compactness,
        )

        completed_small = superpixel_plane_complete(
            depth=tof,
            labels=labels,
            valid_mask=valid_mask,
            plane_min_points=args.plane_min_points,
            plane_max_residual=args.plane_max_residual,
            plane_max_gradient=args.plane_max_gradient,
            plane_blend_valid=args.plane_blend_valid,
            plane_fill_missing_only=args.plane_fill_missing_only,
        )
        completed_small = postprocess(completed_small, args.post_median_ksize)

        completed_up = cv2.resize(
            completed_small,
            (rgb.shape[1], rgb.shape[0]),
            interpolation=interpolation,
        )

        cv2.imwrite(str(output_dir / f"{stem}.png"), completed_small)
        cv2.imwrite(str(upsampled_output_dir / f"{stem}.png"), completed_up)
        print(f"[{index:03d}/{len(stems):03d}] {stem}")

    print("Done.")


if __name__ == "__main__":
    main()
