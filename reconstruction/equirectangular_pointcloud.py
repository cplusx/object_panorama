from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh

from inverse_spherical_representation import equirectangular_direction_map


EDGE_HIT_COLOR_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 72, 72),
    (255, 196, 0),
    (0, 208, 255),
    (116, 92, 255),
    (72, 232, 160),
)
PRED_EDGE_COLOR_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (0, 208, 255),
    (64, 156, 255),
    (128, 200, 255),
)
TARGET_EDGE_COLOR_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 120, 72),
    (255, 176, 64),
    (255, 224, 120),
)


def build_direction_map(height: int) -> np.ndarray:
    return equirectangular_direction_map(int(height), device="cpu").cpu().numpy().astype(np.float32)


def decode_edge_points(
    edge_depth: np.ndarray,
    resolution: int | None = None,
    color_palette_rgb: tuple[tuple[int, int, int], ...] = EDGE_HIT_COLOR_PALETTE_RGB,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    edge_depth = np.asarray(edge_depth, dtype=np.float32)
    if edge_depth.ndim != 3:
        raise ValueError(f"Expected edge_depth with shape [hits, H, W], got {tuple(edge_depth.shape)}")
    directions = build_direction_map(int(resolution or edge_depth.shape[1]))

    point_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    point_counts_per_hit: list[int] = []
    for hit_index in range(edge_depth.shape[0]):
        depth_hit = edge_depth[hit_index]
        valid_mask = np.isfinite(depth_hit) & (depth_hit > 1e-8)
        hit_count = int(valid_mask.sum())
        point_counts_per_hit.append(hit_count)
        if not np.any(valid_mask):
            continue
        points = directions[valid_mask] * depth_hit[valid_mask, None].astype(np.float32)
        hit_color = np.asarray(color_palette_rgb[hit_index % len(color_palette_rgb)], dtype=np.float32) / 255.0
        colors = np.repeat(hit_color[None, :], repeats=hit_count, axis=0)
        point_blocks.append(points.astype(np.float32))
        color_blocks.append(colors.astype(np.float32))

    if not point_blocks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), point_counts_per_hit
    return np.concatenate(point_blocks, axis=0), np.concatenate(color_blocks, axis=0), point_counts_per_hit


def export_point_cloud(points: np.ndarray, colors_rgb: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.PointCloud(vertices=np.asarray(points, dtype=np.float32), colors=_to_rgba_uint8(colors_rgb))
    cloud.export(str(output_path))


def export_overlap_pointcloud_glb(
    model_points: np.ndarray,
    model_colors: np.ndarray,
    edge_points: np.ndarray,
    edge_colors: np.ndarray,
    output_path: Path,
    model_node_name: str = "model_points",
    edge_node_name: str = "edge_points",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    if len(model_points) > 0:
        scene.add_geometry(
            trimesh.PointCloud(vertices=np.asarray(model_points, dtype=np.float32), colors=_to_rgba_uint8(model_colors)),
            node_name=model_node_name,
        )
    if len(edge_points) > 0:
        scene.add_geometry(
            trimesh.PointCloud(vertices=np.asarray(edge_points, dtype=np.float32), colors=_to_rgba_uint8(edge_colors)),
            node_name=edge_node_name,
        )
    scene.export(str(output_path))


def save_edge_depth_comparison_pointclouds(
    output_dir: str | Path,
    pred_edge_depth: torch.Tensor | np.ndarray,
    target_edge_depth: torch.Tensor | np.ndarray,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pred_edge_depth_np = _to_edge_depth_numpy(pred_edge_depth)
    target_edge_depth_np = _to_edge_depth_numpy(target_edge_depth)
    resolution = int(pred_edge_depth_np.shape[1])

    pred_points, pred_colors, _ = decode_edge_points(
        pred_edge_depth_np,
        resolution=resolution,
        color_palette_rgb=PRED_EDGE_COLOR_PALETTE_RGB,
    )
    target_points, target_colors, _ = decode_edge_points(
        target_edge_depth_np,
        resolution=resolution,
        color_palette_rgb=TARGET_EDGE_COLOR_PALETTE_RGB,
    )

    pred_path = output_path / "pred_edge_points.ply"
    target_path = output_path / "target_edge_points.ply"
    overlap_path = output_path / "overlap_pointcloud.glb"
    export_point_cloud(pred_points, pred_colors, pred_path)
    export_point_cloud(target_points, target_colors, target_path)
    export_overlap_pointcloud_glb(
        model_points=target_points,
        model_colors=target_colors,
        edge_points=pred_points,
        edge_colors=pred_colors,
        output_path=overlap_path,
        model_node_name="target_edge_points",
        edge_node_name="pred_edge_points",
    )
    return {
        "pred_edge_points": str(pred_path),
        "target_edge_points": str(target_path),
        "overlap_pointcloud": str(overlap_path),
    }


def _to_rgba_uint8(colors_rgb: np.ndarray) -> np.ndarray:
    colors_rgb = np.clip(np.asarray(colors_rgb, dtype=np.float32), 0.0, 1.0)
    alpha = np.ones((colors_rgb.shape[0], 1), dtype=np.float32)
    rgba = np.concatenate([colors_rgb, alpha], axis=1)
    return np.round(rgba * 255.0).astype(np.uint8)


def _to_edge_depth_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().to(dtype=torch.float32).numpy()
    edge_depth = np.asarray(values, dtype=np.float32)
    if edge_depth.ndim == 4:
        if edge_depth.shape[0] != 1:
            raise ValueError(f"Expected a single-sample edge depth tensor, got {tuple(edge_depth.shape)}")
        edge_depth = edge_depth[0]
    if edge_depth.ndim != 3:
        raise ValueError(f"Expected edge depth with shape [hits, H, W], got {tuple(edge_depth.shape)}")
    return edge_depth