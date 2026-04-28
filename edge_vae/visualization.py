from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch

from reconstruction.equirectangular_pointcloud import save_edge_depth_comparison_pointclouds


def save_edge_vae_outputs(
    output_dir: str | Path,
    *,
    mode: str,
    target_edge_depth: np.ndarray,
    input_encoded: np.ndarray,
    decoded_encoded: np.ndarray,
    decoded_edge_depth: np.ndarray,
    decode_valid_threshold: float,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = _mode_prefix(mode)
    input_encoded_path = output_path / f"input_{prefix}_encoded.pt"
    decoded_encoded_path = output_path / f"decoded_{prefix}_encoded.pt"
    decoded_edge_depth_path = output_path / "decoded_edge_depth.pt"
    preview_path = output_path / "preview.png"

    torch.save(torch.from_numpy(np.asarray(input_encoded, dtype=np.float32)), input_encoded_path)
    torch.save(torch.from_numpy(np.asarray(decoded_encoded, dtype=np.float32)), decoded_encoded_path)
    torch.save(torch.from_numpy(np.asarray(decoded_edge_depth, dtype=np.float32)), decoded_edge_depth_path)

    save_roundtrip_preview(
        preview_path,
        mode=mode,
        target_edge_depth=target_edge_depth,
        input_encoded=input_encoded,
        decoded_encoded=decoded_encoded,
        decoded_edge_depth=decoded_edge_depth,
        decode_valid_threshold=decode_valid_threshold,
    )
    pointcloud_outputs = save_roundtrip_pointclouds(
        output_path,
        target_edge_depth=target_edge_depth,
        decoded_edge_depth=decoded_edge_depth,
    )
    return {
        f"input_{prefix}_encoded": str(input_encoded_path),
        f"decoded_{prefix}_encoded": str(decoded_encoded_path),
        "decoded_edge_depth": str(decoded_edge_depth_path),
        "preview": str(preview_path),
        **pointcloud_outputs,
    }


def save_roundtrip_preview(
    output_path: str | Path,
    *,
    mode: str,
    target_edge_depth: np.ndarray,
    input_encoded: np.ndarray,
    decoded_encoded: np.ndarray,
    decoded_edge_depth: np.ndarray,
    decode_valid_threshold: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_hit = np.asarray(target_edge_depth[0], dtype=np.float32)
    input_hit = np.asarray(input_encoded[0], dtype=np.float32)
    decoded_encoded_hit = np.asarray(decoded_encoded[0], dtype=np.float32)
    decoded_hit = np.asarray(decoded_edge_depth[0], dtype=np.float32)
    error_hit = np.abs(decoded_hit - target_hit).astype(np.float32)
    threshold_mask = (decoded_encoded_hit > float(decode_valid_threshold)).astype(np.float32)

    depth_vmax = max(float(np.nanmax(target_hit)), float(np.nanmax(decoded_hit)), 1.0e-6)
    error_vmax = max(float(np.nanmax(error_hit)), 1.0e-6)
    encoded_limits = _encoded_display_limits(mode=mode, input_hit=input_hit, decoded_encoded_hit=decoded_encoded_hit)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        (target_hit, "Original Edge Depth hit0", "viridis", 0.0, depth_vmax),
        (input_hit, f"{_mode_prefix(mode).upper()} Encoded Input hit0", encoded_limits["cmap"], encoded_limits["vmin"], encoded_limits["vmax"]),
        (decoded_encoded_hit, f"Decoded {_mode_prefix(mode).upper()} Tensor hit0", encoded_limits["cmap"], encoded_limits["vmin"], encoded_limits["vmax"]),
        (decoded_hit, "Decoded Edge Depth hit0", "viridis", 0.0, depth_vmax),
        (error_hit, "Abs Error hit0", "magma", 0.0, error_vmax),
        (threshold_mask, "Valid / Threshold Mask hit0", "gray", 0.0, 1.0),
    ]

    for axis, (image, title, cmap, vmin, vmax) in zip(axes.flat, panels, strict=True):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_roundtrip_pointclouds(
    output_dir: str | Path,
    *,
    target_edge_depth: np.ndarray,
    decoded_edge_depth: np.ndarray,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    outputs = save_edge_depth_comparison_pointclouds(
        output_path,
        pred_edge_depth=np.asarray(decoded_edge_depth, dtype=np.float32),
        target_edge_depth=np.asarray(target_edge_depth, dtype=np.float32),
    )

    overlap_path = output_path / "overlap_pointcloud.ply"
    shutil.copyfile(outputs["target_pred_points"], overlap_path)
    outputs["overlap_pointcloud"] = str(overlap_path)
    return outputs


def _mode_prefix(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in {"raw", "df"}:
        raise ValueError(f"Unsupported preview mode '{mode}'")
    return normalized


def _encoded_display_limits(mode: str, *, input_hit: np.ndarray, decoded_encoded_hit: np.ndarray) -> dict[str, float | str]:
    normalized = _mode_prefix(mode)
    if normalized == "df":
        return {"cmap": "coolwarm", "vmin": -1.0, "vmax": 1.0}
    vmax = max(float(np.nanmax(input_hit)), float(np.nanmax(decoded_encoded_hit)), 1.0e-6)
    return {"cmap": "viridis", "vmin": 0.0, "vmax": vmax}