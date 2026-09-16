from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline ICP SLAM for calibrated, rectified depth frames."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--depth-dir", type=str, default="tof_wmf_tradeoff_opt")
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--output-dir", type=str, default="slam_icp_output")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Depth units per meter. For millimeters use 1000.",
    )
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--sample-step", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--max-correspondence", type=float, default=0.08)
    parser.add_argument("--icp-iterations", type=int, default=15)
    parser.add_argument("--map-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def list_common_stems(depth_dir: Path, rgb_dir: Path) -> list[str]:
    depth_stems = {path.stem for path in depth_dir.glob("*.png")}
    rgb_stems = {path.stem for path in rgb_dir.glob("*.png")}
    return sorted(depth_stems & rgb_stems, key=lambda name: int(name))


def read_u16(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    if image.dtype != np.uint16:
        raise ValueError(f"Expected uint16 PNG, got {image.dtype} for {path}")
    return image


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def depth_to_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    sample_step: int,
) -> tuple[np.ndarray, np.ndarray]:
    if rgb.shape[:2] != depth.shape:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

    ys, xs = np.mgrid[0 : depth.shape[0] : sample_step, 0 : depth.shape[1] : sample_step]
    z = depth[::sample_step, ::sample_step].astype(np.float32) / depth_scale
    mask = (z >= min_depth) & (z <= max_depth)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    xs = xs[mask].astype(np.float32)
    ys = ys[mask].astype(np.float32)
    z = z[mask]

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb[::sample_step, ::sample_step][mask].astype(np.uint8)
    return points, colors


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int32)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    pts = np.zeros((len(uniq), 3), dtype=np.float32)
    cols = np.zeros((len(uniq), 3), dtype=np.float32)
    counts = np.bincount(inverse)
    for i in range(3):
        pts[:, i] = np.bincount(inverse, weights=points[:, i], minlength=len(uniq)) / counts
        cols[:, i] = np.bincount(inverse, weights=colors[:, i], minlength=len(uniq)) / counts
    return pts, np.clip(np.rint(cols), 0, 255).astype(np.uint8)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    homog = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (homog @ transform.T)[:, :3].astype(np.float32)


def best_fit_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    H = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_centroid - R @ src_centroid
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R.astype(np.float32)
    transform[:3, 3] = t.astype(np.float32)
    return transform


def icp_point_to_point(
    src: np.ndarray,
    dst: np.ndarray,
    max_correspondence: float,
    iterations: int,
    init_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, float, int]:
    transform = np.eye(4, dtype=np.float32) if init_transform is None else init_transform.astype(np.float32).copy()
    if len(src) < 10 or len(dst) < 10:
        return transform, np.inf, 0

    tree = cKDTree(dst)
    mean_error = np.inf
    inliers = 0

    for _ in range(iterations):
        src_transformed = transform_points(src, transform)
        distances, indices = tree.query(src_transformed, k=1)
        mask = distances < max_correspondence
        if int(mask.sum()) < 10:
            break

        matched_src = src_transformed[mask]
        matched_dst = dst[indices[mask]]
        delta = best_fit_transform(matched_src, matched_dst)
        transform = delta @ transform
        mean_error = float(distances[mask].mean())
        inliers = int(mask.sum())

    return transform, mean_error, inliers


def save_trajectory(stems: list[str], poses: list[np.ndarray], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for stem, pose in zip(stems, poses):
            flat = " ".join(f"{value:.8f}" for value in pose.reshape(-1))
            handle.write(f"{stem} {flat}\n")


def save_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    depth_dir = root / args.depth_dir
    rgb_dir = root / args.rgb_dir
    output_dir = root / args.output_dir

    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")

    ensure_dir(output_dir)
    save_run_config(args, output_dir)

    stems = list_common_stems(depth_dir, rgb_dir)
    if args.limit > 0:
        stems = stems[: args.limit]
    if not stems:
        raise RuntimeError("No matching depth/RGB PNG pairs found.")

    frames = []
    for stem in stems:
        depth = read_u16(depth_dir / f"{stem}.png")
        rgb = read_rgb(rgb_dir / f"{stem}.png")
        if rgb.shape[:2] != depth.shape:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        points, colors = depth_to_point_cloud(
            depth=depth,
            rgb=rgb,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            depth_scale=args.depth_scale,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            sample_step=args.sample_step,
        )
        points, colors = voxel_downsample(points, colors, args.voxel_size)
        frames.append({"stem": stem, "points": points, "colors": colors})

    poses = [np.eye(4, dtype=np.float32)]
    pose = np.eye(4, dtype=np.float32)
    log = []

    print(f"Running ICP SLAM on {len(frames)} frames")
    for idx in range(1, len(frames)):
        prev = frames[idx - 1]
        curr = frames[idx]
        delta, mean_error, inliers = icp_point_to_point(
            src=curr["points"],
            dst=prev["points"],
            max_correspondence=args.max_correspondence,
            iterations=args.icp_iterations,
        )
        pose = poses[-1] @ np.linalg.inv(delta)
        poses.append(pose.astype(np.float32))
        log.append(
            {
                "from": prev["stem"],
                "to": curr["stem"],
                "mean_error": mean_error,
                "inliers": inliers,
            }
        )
        print(f"[{idx+1:03d}/{len(frames):03d}] {curr['stem']} inliers={inliers} error={mean_error:.5f}")

    map_points = []
    map_colors = []
    for idx, frame in enumerate(frames[:: args.map_stride]):
        pose_idx = idx * args.map_stride
        transformed = transform_points(frame["points"], poses[pose_idx])
        map_points.append(transformed)
        map_colors.append(frame["colors"])

    if map_points:
        map_points_arr = np.concatenate(map_points, axis=0)
        map_colors_arr = np.concatenate(map_colors, axis=0)
        map_points_arr, map_colors_arr = voxel_downsample(map_points_arr, map_colors_arr, args.voxel_size)
    else:
        map_points_arr = np.empty((0, 3), dtype=np.float32)
        map_colors_arr = np.empty((0, 3), dtype=np.uint8)

    save_trajectory(stems, poses, output_dir / "trajectory.txt")
    save_ply(map_points_arr, map_colors_arr, output_dir / "map.ply")
    (output_dir / "pairwise_icp_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    summary = {
        "num_frames": len(frames),
        "num_map_points": int(len(map_points_arr)),
        "trajectory_path": str((output_dir / "trajectory.txt").resolve()),
        "map_path": str((output_dir / "map.ply").resolve()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
