from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .slam import (
    depth_to_point_cloud,
    icp_point_to_point,
    save_ply,
    save_trajectory,
    transform_points,
    voxel_downsample,
)
from .visual_odometry import (
    list_common_stems,
    load_intrinsics,
    pnp_transform,
    read_rgb,
    read_u16,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature VO with ICP refinement and depth fusion."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--rgb-dir", type=str, default="rgb_rect")
    parser.add_argument("--depth-dir", type=str, default="tof_wmf_tradeoff_opt_upsampled")
    parser.add_argument("--output-dir", type=str, default="feature_vo_icp_fusion_output")
    parser.add_argument("--calibration-file", type=Path, default=Path("calibration_parameters.json"))
    parser.add_argument(
        "--camera",
        choices=["cam_rgb", "cam_tof"],
        default="cam_rgb",
        help="Which intrinsic model to use for the input depth image.",
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
    parser.add_argument("--max-correspondence", type=float, default=0.08)
    parser.add_argument("--icp-iterations", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    config = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


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
    log = []

    print(f"Running feature VO + ICP refinement on {len(frames)} frames")
    for idx in range(1, len(frames)):
        prev = frames[idx - 1]
        curr = frames[idx]

        vo_delta, vo_stats = pnp_transform(
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

        icp_delta, icp_error, icp_inliers = icp_point_to_point(
            src=curr["points"],
            dst=prev["points"],
            max_correspondence=args.max_correspondence,
            iterations=args.icp_iterations,
            init_transform=vo_delta,
        )

        pose = poses[-1] @ np.linalg.inv(icp_delta)
        poses.append(pose.astype(np.float32))

        entry = {
            "from": prev["stem"],
            "to": curr["stem"],
            "vo_num_matches": vo_stats["num_matches"],
            "vo_num_3d_correspondences": vo_stats["num_3d_correspondences"],
            "vo_num_inliers": vo_stats["num_inliers"],
            "vo_reproj_error": vo_stats["reproj_error"],
            "vo_used_identity": vo_stats["used_identity"],
            "icp_inliers": icp_inliers,
            "icp_mean_error": icp_error,
        }
        log.append(entry)
        print(
            f"[{idx+1:03d}/{len(frames):03d}] {curr['stem']} "
            f"vo_matches={entry['vo_num_matches']} vo_inliers={entry['vo_num_inliers']} "
            f"icp_inliers={entry['icp_inliers']} icp_error={entry['icp_mean_error']:.5f} "
            f"vo_identity={entry['vo_used_identity']}"
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
    (output_dir / "pairwise_vo_icp_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

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
