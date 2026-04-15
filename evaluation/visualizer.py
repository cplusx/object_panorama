from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_debug_tensors(output_dir: str | Path, batch: dict, pred: torch.Tensor, target: torch.Tensor) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(batch["input"].detach().cpu(), output_path / "input.pt")
    torch.save(batch["condition"].detach().cpu(), output_path / "condition.pt")
    torch.save(target.detach().cpu(), output_path / "target.pt")
    torch.save(pred.detach().cpu(), output_path / "pred.pt")


def save_preview_png(output_dir: str | Path, batch: dict, pred: torch.Tensor, target: torch.Tensor) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_panel = _to_preview_panel(batch["input"])
    target_panel = _to_preview_panel(target)
    pred_panel = _to_preview_panel(pred)
    canvas = torch.cat([input_panel, target_panel, pred_panel], dim=-1)
    image = (canvas.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(output_path / "preview.png")


def _to_preview_panel(tensor: torch.Tensor) -> torch.Tensor:
    panel = tensor.detach().cpu()
    if panel.ndim == 4:
        panel = panel[0]
    if panel.ndim != 3:
        raise ValueError(f"Expected tensor with shape [B, C, H, W] or [C, H, W], got {tuple(tensor.shape)}")
    if panel.shape[0] == 1:
        panel = panel.repeat(3, 1, 1)
    elif panel.shape[0] == 2:
        panel = torch.cat([panel, panel[:1]], dim=0)
    else:
        panel = panel[:3]

    panel_min = panel.min()
    panel_max = panel.max()
    if float(panel_max - panel_min) < 1e-8:
        return torch.zeros_like(panel)
    return (panel - panel_min) / (panel_max - panel_min)