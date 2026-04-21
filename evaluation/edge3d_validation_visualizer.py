from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from reconstruction import save_model_target_pred_pointclouds


def _normalize_panel(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    valid = values[finite]
    lo = float(valid.min())
    hi = float(valid.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((np.nan_to_num(values, nan=lo) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def save_edge3d_validation_preview(
    output_dir,
    batch,
    pred_edge_depth,
    sample_ids=None,
    max_items=None,
    save_reconstruction=False,
) -> list[dict[str, object]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_depth = torch.nan_to_num(batch["model_depth"], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu()
    target_edge_depth = torch.nan_to_num(batch["edge_depth"], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu()
    pred_edge_depth = pred_edge_depth.detach().cpu()
    resolved_sample_ids = list(sample_ids or batch.get("sample_ids") or [])
    if not resolved_sample_ids:
        resolved_sample_ids = [f"sample_{index:06d}" for index in range(int(pred_edge_depth.shape[0]))]
    limit = len(resolved_sample_ids) if max_items is None else min(int(max_items), len(resolved_sample_ids))
    saved_outputs: list[dict[str, object]] = []

    for index in range(limit):
        sample_dir = output_path / str(resolved_sample_ids[index])
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_model = model_depth[index : index + 1]
        sample_pred = pred_edge_depth[index : index + 1]
        sample_target = target_edge_depth[index : index + 1]

        torch.save(sample_pred, sample_dir / "pred_edge_depth.pt")
        torch.save(sample_target, sample_dir / "target_edge_depth.pt")

        target_hit0 = sample_target[0, 0].numpy()
        pred_hit0 = sample_pred[0, 0].numpy()
        error_hit0 = np.abs(target_hit0 - pred_hit0)

        figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        panels = [
            (_normalize_panel(target_hit0), "target edge depth hit0"),
            (_normalize_panel(pred_hit0), "pred edge depth hit0"),
            (_normalize_panel(error_hit0), "abs error hit0"),
        ]
        for axis, (image, title) in zip(axes, panels):
            axis.imshow(image, cmap="magma")
            axis.set_title(title)
            axis.axis("off")
        preview_path = sample_dir / "preview.png"
        figure.savefig(preview_path, dpi=160)
        plt.close(figure)

        reconstruction_outputs: dict[str, str] = {}
        if save_reconstruction:
            reconstruction_outputs = save_model_target_pred_pointclouds(
                sample_dir,
                model_depth=sample_model,
                pred_edge_depth=sample_pred,
                target_edge_depth=sample_target,
            )

        saved_outputs.append(
            {
                "sample_id": str(resolved_sample_ids[index]),
                "sample_dir": sample_dir,
                "preview_path": preview_path,
                "reconstruction_outputs": reconstruction_outputs,
            }
        )

    return saved_outputs