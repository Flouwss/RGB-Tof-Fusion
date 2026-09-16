from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .slam import (
    depth_to_point_cloud,
    save_ply,
    save_trajectory,
    transform_points,
    voxel_downsample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline RGB visual odometry with depth fusion."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--depth-dir", type=str, default="tof_wmf_tradeoff_opt_upsampled")
    parser.add_argument("--output-dir", type=str, default="vo_depth_fusion_output")
    parser.add_argument("--calibration-file", type=Path, default=Path("calibration_parameters.json"))
    parser.add_argument(
        "--camera",
        choices=["cam_rgb", "cam_tof"],
        default="cam_rgb",
        help="Which intrinsic model to use for the depth image.",
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--sample-step", type=int, default=8)
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--map-stride", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--max-reproj-error", type=float, default=4.0)
    parser.add_argument("--min-correspondences", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


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


def load_intrinsics(calibration_file: Path, camera: str) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(calibration_file.read_text(encoding="utf-8"))
    intr = data["intrinsic"][camera]
    cam_mat = np.asarray(intr["cam_mat"], dtype=np.float32)
    dist = np.asarray(intr["dist"], dtype=np.float32)
    return cam_mat, dist


def list_common_stems(depth_dir: Path, rgb_dir: Path) -> list[str]:
    depth_stems = {path.stem for path in depth_dir.glob("*.png")}
    rgb_stems = {path.stem for path in rgb_dir.glob("*.png")}
    return sorted(depth_stems & rgb_stems, key=lambda name: int(name))


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


def pnp_transform(
    prev_rgb: np.ndarray,
    prev_depth: np.ndarray,
    curr_rgb: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    max_features: int,
    ratio_test: float,
    max_reproj_error: float,
    min_correspondences: int,
) -> tuple[np.ndarray, dict[str, float]]:
    gray_prev = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp_prev, des_prev = orb.detectAndCompute(gray_prev, None)
    kp_curr, des_curr = orb.detectAndCompute(gray_curr, None)

    if des_prev is None or des_curr is None or len(kp_prev) == 0 or len(kp_curr) == 0:
        return np.eye(4, dtype=np.float32), {
            "num_matches": 0,
            "num_3d_correspondences": 0,
            "num_inliers": 0,
            "reproj_error": float("inf"),
            "used_identity": 1,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_prev, des_curr, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good.append(m)

    if len(good) < min_correspondences:
        return np.eye(4, dtype=np.float32), {
            "num_matches": len(good),
            "num_3d_correspondences": 0,
            "num_inliers": 0,
            "reproj_error": float("inf"),
            "used_identity": 1,
        }

    object_points = []
    image_points = []
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    for match in good:
        u_prev, v_prev = kp_prev[match.queryIdx].pt
        u_curr, v_curr = kp_curr[match.trainIdx].pt
        x = int(round(u_prev))
        y = int(round(v_prev))
        if not (0 <= x < prev_depth.shape[1] and 0 <= y < prev_depth.shape[0]):
            continue
        z = float(prev_depth[y, x]) / depth_scale
        if z < min_depth or z > max_depth:
            continue
        X = (u_prev - cx) * z / fx
        Y = (v_prev - cy) * z / fy
        object_points.append([X, Y, z])
        image_points.append([u_curr, v_curr])

    if len(object_points) < min_correspondences:
        return np.eye(4, dtype=np.float32), {
            "num_matches": len(good),
            "num_3d_correspondences": len(object_points),
            "num_inliers": 0,
            "reproj_error": float("inf"),
            "used_identity": 1,
        }

    object_points_arr = np.asarray(object_points, dtype=np.float32)
    image_points_arr = np.asarray(image_points, dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points_arr,
        image_points_arr,
        K,
        dist,
        reprojectionError=max_reproj_error,
        iterationsCount=100,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None or len(inliers) < min_correspondences:
        return np.eye(4, dtype=np.float32), {
            "num_matches": len(good),
            "num_3d_correspondences": len(object_points),
            "num_inliers": 0,
            "reproj_error": float("inf"),
            "used_identity": 1,
        }

    inlier_obj = object_points_arr[inliers[:, 0]]
    inlier_img = image_points_arr[inliers[:, 0]]
    success, rvec, tvec = cv2.solvePnP(
        inlier_obj,
        inlier_img,
        K,
        dist,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return np.eye(4, dtype=np.float32), {
            "num_matches": len(good),
            "num_3d_correspondences": len(object_points),
            "num_inliers": len(inliers),
            "reproj_error": float("inf"),
            "used_identity": 1,
        }

    R, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R.astype(np.float32)
    transform[:3, 3] = tvec[:, 0].astype(np.float32)

    projected, _ = cv2.projectPoints(inlier_obj, rvec, tvec, K, dist)
    projected = projected[:, 0, :]
    reproj_error = float(np.mean(np.linalg.norm(projected - inlier_img, axis=1)))

    return transform, {
        "num_matches": len(good),
        "num_3d_correspondences": len(object_points),
        "num_inliers": int(len(inliers)),
        "reproj_error": reproj_error,
        "used_identity": 0,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rgb_dir = root / args.rgb_dir
    depth_dir = root / args.depth_dir
    output_dir = root / args.output_dir
    calib_file = (root / args.calibration_file).resolve()

    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
    if not calib_file.is_file():
        raise FileNotFoundError(f"Missing calibration file: {calib_file}")

    ensure_dir(output_dir)
    save_run_config(args, output_dir)

    K, dist = load_intrinsics(calib_file, args.camera)
    stems = list_common_stems(depth_dir, rgb_dir)
    if args.limit > 0:
        stems = stems[: args.limit]
    if not stems:
        raise RuntimeError("No matching RGB/depth PNG pairs found.")

    frames = []
    for stem in stems:
        rgb = read_rgb(rgb_dir / f"{stem}.png")
        depth = read_u16(depth_dir / f"{stem}.png")
        if rgb.shape[:2] != depth.shape:
            raise ValueError(
                f"RGB/depth size mismatch for frame {stem}: rgb {rgb.shape[:2]}, depth {depth.shape}"
            )
        points, colors = depth_to_point_cloud(
            depth=depth,
            rgb=rgb,
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            depth_scale=args.depth_scale,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            sample_step=args.sample_step,
        )
        points, colors = voxel_downsample(points, colors, args.voxel_size)
        frames.append({"stem": stem, "rgb": rgb, "depth": depth, "points": points, "colors": colors})

    poses = [np.eye(4, dtype=np.float32)]
    pose = np.eye(4, dtype=np.float32)
    log = []

    print(f"Running visual odometry + depth fusion on {len(frames)} frames")
    for idx in range(1, len(frames)):
        prev = frames[idx - 1]
        curr = frames[idx]
        delta, stats = pnp_transform(
            prev_rgb=prev["rgb"],
            prev_depth=prev["depth"],
            curr_rgb=curr["rgb"],
            K=K,
            dist=dist,
            depth_scale=args.depth_scale,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            max_features=args.max_features,
            ratio_test=args.ratio_test,
            max_reproj_error=args.max_reproj_error,
            min_correspondences=args.min_correspondences,
        )
        pose = poses[-1] @ np.linalg.inv(delta)
        poses.append(pose.astype(np.float32))
        stats["from"] = prev["stem"]
        stats["to"] = curr["stem"]
        log.append(stats)
        print(
            f"[{idx+1:03d}/{len(frames):03d}] {curr['stem']} "
            f"matches={stats['num_matches']} 3d={stats['num_3d_correspondences']} "
            f"inliers={stats['num_inliers']} reproj={stats['reproj_error']:.3f} "
            f"identity={stats['used_identity']}"
        )

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
    (output_dir / "pairwise_vo_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    summary = {
        "num_frames": len(frames),
        "num_map_points": int(len(map_points_arr)),
        "trajectory_path": str((output_dir / "trajectory.txt").resolve()),
        "map_path": str((output_dir / "map.ply").resolve()),
        "camera": args.camera,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
