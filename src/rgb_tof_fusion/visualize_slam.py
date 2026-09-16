from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SLAM map and trajectory as an SVG with orthographic projections."
    )
    parser.add_argument("--slam-dir", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, default=None)
    parser.add_argument("--point-size", type=float, default=1.2)
    parser.add_argument("--traj-width", type=float, default=2.0)
    parser.add_argument("--padding", type=float, default=24.0)
    parser.add_argument("--panel-size", type=int, default=420)
    parser.add_argument("--max-points", type=int, default=5000)
    return parser.parse_args()


def read_ply_vertices(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count = None
    end_header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
        if line.strip() == "end_header":
            end_header_idx = idx
            break
    if vertex_count is None or end_header_idx is None:
        raise ValueError(f"Invalid PLY header: {path}")

    points = []
    colors = []
    for line in lines[end_header_idx + 1 : end_header_idx + 1 + vertex_count]:
        parts = line.split()
        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)


def read_trajectory(path: Path) -> tuple[list[str], np.ndarray]:
    stems: list[str] = []
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        stems.append(parts[0])
        mat = np.asarray([float(x) for x in parts[1:]], dtype=np.float32).reshape(4, 4)
        poses.append(mat)
    if not poses:
        raise RuntimeError(f"No poses found in {path}")
    return stems, np.stack(poses, axis=0)


def subsample_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int32)
    return points[idx], colors[idx]


def project_points(
    points_2d: np.ndarray,
    width: float,
    height: float,
    padding: float,
) -> np.ndarray:
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    scale = min((width - 2 * padding) / spans[0], (height - 2 * padding) / spans[1])
    centered = (points_2d - mins) * scale
    offset_x = (width - centered[:, 0].max() - centered[:, 0].min()) * 0.5
    offset_y = (height - centered[:, 1].max() - centered[:, 1].min()) * 0.5
    out = np.empty_like(centered)
    out[:, 0] = centered[:, 0] + offset_x
    out[:, 1] = height - (centered[:, 1] + offset_y)
    return out


def rgb_hex(color: np.ndarray) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def build_panel(
    title: str,
    axis_labels: tuple[str, str],
    map_points: np.ndarray,
    map_colors: np.ndarray,
    traj_points: np.ndarray,
    width: float,
    height: float,
    padding: float,
    point_size: float,
    traj_width: float,
    x_offset: float,
    y_offset: float,
) -> str:
    all_points = np.concatenate([map_points, traj_points], axis=0) if len(traj_points) else map_points
    projected_all = project_points(all_points, width, height, padding)
    projected_map = projected_all[: len(map_points)]
    projected_traj = projected_all[len(map_points) :]

    parts = [
        f'<g transform="translate({x_offset},{y_offset})">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa" stroke="#333" stroke-width="1"/>',
        f'<text x="{padding}" y="{padding}" font-size="18" font-family="Arial" fill="#111">{title}</text>',
        f'<text x="{width - padding - 18}" y="{height - 8}" font-size="14" font-family="Arial" fill="#444">{axis_labels[0]}</text>',
        f'<text x="10" y="{padding + 18}" font-size="14" font-family="Arial" fill="#444">{axis_labels[1]}</text>',
    ]

    for point, color in zip(projected_map, map_colors):
        parts.append(
            f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="{point_size}" fill="{rgb_hex(color)}" fill-opacity="0.75"/>'
        )

    if len(projected_traj):
        path = " ".join(
            [f"M {projected_traj[0,0]:.2f} {projected_traj[0,1]:.2f}"]
            + [f"L {p[0]:.2f} {p[1]:.2f}" for p in projected_traj[1:]]
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="#d62828" stroke-width="{traj_width}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for idx, point in enumerate(projected_traj):
            radius = 4.0 if idx in (0, len(projected_traj) - 1) else 2.2
            color = "#2a9d8f" if idx == 0 else "#d62828"
            parts.append(
                f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" r="{radius}" fill="{color}" stroke="#fff" stroke-width="0.6"/>'
            )

    parts.append("</g>")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    slam_dir = args.slam_dir.resolve()
    if not slam_dir.is_dir():
        raise FileNotFoundError(f"Missing SLAM directory: {slam_dir}")

    map_path = slam_dir / "map.ply"
    traj_path = slam_dir / "trajectory.txt"
    summary_path = slam_dir / "summary.json"
    if not map_path.is_file() or not traj_path.is_file():
        raise FileNotFoundError("Expected map.ply and trajectory.txt in SLAM directory.")

    points, colors = read_ply_vertices(map_path)
    points, colors = subsample_points(points, colors, args.max_points)
    stems, poses = read_trajectory(traj_path)
    traj = poses[:, :3, 3]

    output_svg = args.output_svg.resolve() if args.output_svg else slam_dir / "visualization.svg"

    panel = float(args.panel_size)
    gap = 24.0
    width = panel * 3 + gap * 4
    height = panel + gap * 2

    xy_map = points[:, [0, 1]]
    xz_map = points[:, [0, 2]]
    yz_map = points[:, [1, 2]]
    xy_traj = traj[:, [0, 1]]
    xz_traj = traj[:, [0, 2]]
    yz_traj = traj[:, [1, 2]]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f1f3f5"/>',
        build_panel(
            "XY Projection",
            ("X", "Y"),
            xy_map,
            colors,
            xy_traj,
            panel,
            panel,
            args.padding,
            args.point_size,
            args.traj_width,
            gap,
            gap,
        ),
        build_panel(
            "XZ Projection",
            ("X", "Z"),
            xz_map,
            colors,
            xz_traj,
            panel,
            panel,
            args.padding,
            args.point_size,
            args.traj_width,
            gap * 2 + panel,
            gap,
        ),
        build_panel(
            "YZ Projection",
            ("Y", "Z"),
            yz_map,
            colors,
            yz_traj,
            panel,
            panel,
            args.padding,
            args.point_size,
            args.traj_width,
            gap * 3 + panel * 2,
            gap,
        ),
        "</svg>",
    ]
    output_svg.write_text("\n".join(svg_parts), encoding="utf-8")

    payload = {
        "slam_dir": str(slam_dir),
        "output_svg": str(output_svg),
        "num_map_points_visualized": int(len(points)),
        "num_poses": int(len(stems)),
    }
    if summary_path.is_file():
        payload["slam_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    (slam_dir / "visualization_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
