from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .depth_completion import (
    confidence_aware_fill,
    normalize_confidence,
    read_rgb,
    read_u16,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confidence-aware completion followed by local plane refinement."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--tof-dir", type=str, default="tof_rect")
    parser.add_argument("--amplitude-dir", type=str, default="amplitude_rect")
    parser.add_argument("--output-dir", type=str, default="tof_confidence_plane")
    parser.add_argument(
        "--upsampled-output-dir",
        type=str,
        default="tof_confidence_plane_upsampled",
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

    parser.add_argument("--plane-radius", type=int, default=2)
    parser.add_argument("--plane-min-points", type=int, default=8)
    parser.add_argument("--plane-max-residual", type=float, default=25.0)
    parser.add_argument("--plane-max-gradient", type=float, default=60.0)
    parser.add_argument("--plane-blend", type=float, default=0.35)
    parser.add_argument("--rgb-edge-threshold", type=float, default=12.0)
    parser.add_argument(
        "--plane-new-only",
        action="store_true",
        help="Refine only pixels that were filled by the completion stage.",
    )
    parser.add_argument(
        "--plane-local-depth-threshold",
        type=float,
        default=45.0,
        help="Skip plane fitting if local depth variation exceeds this threshold.",
    )

    parser.add_argument(
        "--upsample-mode",
        choices=["nearest", "linear"],
        default="nearest",
    )
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


def plane_refine_depth(
    depth: np.ndarray,
    original_valid_mask: np.ndarray,
    rgb_small: np.ndarray,
    radius: int,
    min_points: int,
    max_residual: float,
    max_gradient: float,
    blend: float,
    rgb_edge_threshold: float,
    new_only: bool,
    local_depth_threshold: float,
) -> np.ndarray:
    h, w = depth.shape
    depth_f = depth.astype(np.float32)
    refined = depth_f.copy()

    gray = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb_grad = np.sqrt(gx * gx + gy * gy)

    for y in range(h):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        for x in range(w):
            if new_only and original_valid_mask[y, x]:
                continue
            if rgb_grad[y, x] > rgb_edge_threshold:
                continue

            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)
            patch = depth_f[y0:y1, x0:x1]
            mask = patch > 0
            if int(mask.sum()) < min_points:
                continue
            if float(patch[mask].max() - patch[mask].min()) > local_depth_threshold:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1]
            xx = xx[mask].astype(np.float32)
            yy = yy[mask].astype(np.float32)
            zz = patch[mask].astype(np.float32)

            A = np.stack([xx, yy, np.ones_like(xx)], axis=1)
            coeffs, _, _, _ = np.linalg.lstsq(A, zz, rcond=None)
            a, b, c = coeffs
            pred = a * xx + b * yy + c
            residual = np.abs(pred - zz)
            grad_mag = float(np.sqrt(a * a + b * b))

            if float(np.median(residual)) > max_residual:
                continue
            if grad_mag > max_gradient:
                continue

            plane_value = a * x + b * y + c
            refined[y, x] = (1.0 - blend) * depth_f[y, x] + blend * plane_value

    refined = np.clip(refined, 0, np.iinfo(np.uint16).max)
    return np.rint(refined).astype(np.uint16)


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
        "plane: "
        f"radius={args.plane_radius}, min_points={args.plane_min_points}, max_residual={args.plane_max_residual}, "
        f"max_gradient={args.plane_max_gradient}, blend={args.plane_blend}, rgb_edge_threshold={args.rgb_edge_threshold}, "
        f"new_only={args.plane_new_only}, local_depth_threshold={args.plane_local_depth_threshold}"
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

        refined_small = plane_refine_depth(
            depth=completed_small,
            original_valid_mask=valid_mask,
            rgb_small=rgb_small,
            radius=args.plane_radius,
            min_points=args.plane_min_points,
            max_residual=args.plane_max_residual,
            max_gradient=args.plane_max_gradient,
            blend=args.plane_blend,
            rgb_edge_threshold=args.rgb_edge_threshold,
            new_only=args.plane_new_only,
            local_depth_threshold=args.plane_local_depth_threshold,
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
