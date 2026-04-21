from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_debug_tensors(
    output_dir: str | Path,
    sample: torch.Tensor,
    condition: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(sample.detach().cpu(), output_path / "sample.pt")
    torch.save(condition.detach().cpu(), output_path / "condition.pt")
    torch.save(target.detach().cpu(), output_path / "target.pt")
    torch.save(pred.detach().cpu(), output_path / "pred.pt")


def save_preview_png(output_dir: str | Path, sample: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sample_panel = _to_preview_panel(sample)
    pred_panel = _to_preview_panel(pred)
    target_panel = _to_preview_panel(target)
    error_panel = _to_preview_panel(torch.abs(_to_preview_tensor(pred) - _to_preview_tensor(target)))
    canvas = torch.cat([sample_panel, pred_panel, target_panel, error_panel], dim=-1)
    image = (canvas.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    preview_path = output_path / "preview.png"
    Image.fromarray(image).save(preview_path)
    return preview_path


def _to_preview_panel(tensor: torch.Tensor) -> torch.Tensor:
    panel = _to_preview_tensor(tensor)
    panel_min = panel.min()
    panel_max = panel.max()
    if float(panel_max - panel_min) < 1e-8:
        return torch.zeros((3, panel.shape[1], panel.shape[2]), dtype=panel.dtype)
    normalized = (panel - panel_min) / (panel_max - panel_min)
    return normalized.repeat(3, 1, 1)


def _to_preview_tensor(tensor: torch.Tensor) -> torch.Tensor:
    panel = tensor.detach().cpu()
    if panel.ndim == 4:
        panel = panel[0]
    if panel.ndim != 3:
        raise ValueError(f"Expected tensor with shape [B, C, H, W] or [C, H, W], got {tuple(tensor.shape)}")
    return panel[:1]