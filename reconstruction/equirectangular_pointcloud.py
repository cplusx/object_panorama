from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import trimesh

from edge3d.representation.layered_spherical_representation import equirectangular_direction_map


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
MODEL_SURFACE_COLOR_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (176, 176, 176),
    (156, 156, 156),
    (136, 136, 136),
    (116, 116, 116),
    (96, 96, 96),
)


def build_direction_map(height: int) -> np.ndarray:
    return equirectangular_direction_map(int(height), device="cpu").cpu().numpy().astype(np.float32)


def decode_depth_layers_to_points(
    depth_layers: np.ndarray,
    resolution: int | None = None,
    color_palette_rgb: tuple[tuple[int, int, int], ...] = EDGE_HIT_COLOR_PALETTE_RGB,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    depth_layers = np.asarray(depth_layers, dtype=np.float32)
    if depth_layers.ndim != 3:
        raise ValueError(f"Expected depth_layers with shape [layers, H, W], got {tuple(depth_layers.shape)}")
    directions = build_direction_map(int(resolution or depth_layers.shape[1]))

    point_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    point_counts_per_layer: list[int] = []
    for layer_index in range(depth_layers.shape[0]):
        depth_layer = depth_layers[layer_index]
        valid_mask = np.isfinite(depth_layer) & (depth_layer > 1e-8)
        layer_count = int(valid_mask.sum())
        point_counts_per_layer.append(layer_count)
        if not np.any(valid_mask):
            continue
        points = directions[valid_mask] * depth_layer[valid_mask, None].astype(np.float32)
        layer_color = np.asarray(color_palette_rgb[layer_index % len(color_palette_rgb)], dtype=np.float32) / 255.0
        colors = np.repeat(layer_color[None, :], repeats=layer_count, axis=0)
        point_blocks.append(points.astype(np.float32))
        color_blocks.append(colors.astype(np.float32))

    if not point_blocks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), point_counts_per_layer
    return np.concatenate(point_blocks, axis=0), np.concatenate(color_blocks, axis=0), point_counts_per_layer


def decode_edge_points(
    edge_depth: np.ndarray,
    resolution: int | None = None,
    color_palette_rgb: tuple[tuple[int, int, int], ...] = EDGE_HIT_COLOR_PALETTE_RGB,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    return decode_depth_layers_to_points(
        edge_depth,
        resolution=resolution,
        color_palette_rgb=color_palette_rgb,
    )


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
    export_named_pointclouds_glb(
        [
            (model_node_name, model_points, model_colors),
            (edge_node_name, edge_points, edge_colors),
        ],
        output_path,
    )


def export_named_pointclouds_glb(
    pointclouds: Iterable[tuple[str, np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for node_name, points, colors_rgb in pointclouds:
        if len(points) == 0:
            continue
        scene.add_geometry(
            trimesh.PointCloud(vertices=np.asarray(points, dtype=np.float32), colors=_to_rgba_uint8(colors_rgb)),
            node_name=node_name,
        )
    scene.export(str(output_path))


def save_edge_depth_comparison_pointclouds(
    output_dir: str | Path,
    pred_edge_depth: torch.Tensor | np.ndarray,
    target_edge_depth: torch.Tensor | np.ndarray,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pred_edge_depth_np = _to_depth_layers_numpy(pred_edge_depth, expected_name="pred_edge_depth")
    target_edge_depth_np = _to_depth_layers_numpy(target_edge_depth, expected_name="target_edge_depth")
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


def save_model_target_pred_pointclouds(
    output_dir: str | Path,
    model_depth: torch.Tensor | np.ndarray,
    pred_edge_depth: torch.Tensor | np.ndarray,
    target_edge_depth: torch.Tensor | np.ndarray,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_depth_np = _to_depth_layers_numpy(model_depth, expected_name="model_depth")
    pred_edge_depth_np = _to_depth_layers_numpy(pred_edge_depth, expected_name="pred_edge_depth")
    target_edge_depth_np = _to_depth_layers_numpy(target_edge_depth, expected_name="target_edge_depth")
    resolution = int(model_depth_np.shape[1])

    outputs = save_edge_depth_comparison_pointclouds(
        output_path,
        pred_edge_depth=pred_edge_depth_np,
        target_edge_depth=target_edge_depth_np,
    )

    model_points, model_colors, _ = decode_depth_layers_to_points(
        model_depth_np,
        resolution=resolution,
        color_palette_rgb=MODEL_SURFACE_COLOR_PALETTE_RGB,
    )
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

    model_path = output_path / "model_points.ply"
    overlap_model_target_pred_path = output_path / "overlap_model_target_pred.glb"
    export_point_cloud(model_points, model_colors, model_path)
    export_named_pointclouds_glb(
        [
            ("model_points", model_points, model_colors),
            ("target_edge_points", target_points, target_colors),
            ("pred_edge_points", pred_points, pred_colors),
        ],
        overlap_model_target_pred_path,
    )

    outputs.update(
        {
            "model_points": str(model_path),
            "overlap_model_target_pred": str(overlap_model_target_pred_path),
        }
    )
    return outputs


def _to_rgba_uint8(colors_rgb: np.ndarray) -> np.ndarray:
    colors_rgb = np.clip(np.asarray(colors_rgb, dtype=np.float32), 0.0, 1.0)
    alpha = np.ones((colors_rgb.shape[0], 1), dtype=np.float32)
    rgba = np.concatenate([colors_rgb, alpha], axis=1)
    return np.round(rgba * 255.0).astype(np.uint8)


def _to_depth_layers_numpy(values: torch.Tensor | np.ndarray, expected_name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().to(dtype=torch.float32).numpy()
    depth_layers = np.asarray(values, dtype=np.float32)
    if depth_layers.ndim == 4:
        if depth_layers.shape[0] != 1:
            raise ValueError(f"Expected a single-sample {expected_name} tensor, got {tuple(depth_layers.shape)}")
        depth_layers = depth_layers[0]
    if depth_layers.ndim != 3:
        raise ValueError(f"Expected {expected_name} with shape [layers, H, W], got {tuple(depth_layers.shape)}")
    return depth_layers