from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .types import ModelInputBatch


def build_jit_flow_matching_batch(
    batch: dict[str, Any],
    t_min: float,
    t_max: float,
    noise_scale: float = 1.0,
    condition_type_id: int = 0,
) -> ModelInputBatch:
    x0 = torch.nan_to_num(batch["edge_depth"], nan=0.0, posinf=0.0, neginf=0.0)
    if x0.shape[1] != 3:
        raise ValueError(f"Expected edge_depth with 3 hits/channels, got {tuple(x0.shape)}")

    noise = torch.randn_like(x0) * float(noise_scale)
    timestep = torch.empty(x0.shape[0], device=x0.device, dtype=x0.dtype).uniform_(float(t_min), float(t_max))
    timestep_view = timestep.view(-1, 1, 1, 1)
    x_t = (1.0 - timestep_view) * x0 + timestep_view * noise

    condition = torch.cat(
        [
            torch.nan_to_num(batch["model_rgb"], nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(batch["model_depth"], nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(batch["model_normal"], nan=0.0, posinf=0.0, neginf=0.0),
        ],
        dim=1,
    )
    condition_type_ids = torch.full(
        (x0.shape[0],),
        int(condition_type_id),
        device=x0.device,
        dtype=torch.long,
    )

    return ModelInputBatch(
        sample=x_t,
        condition=condition,
        condition_type_ids=condition_type_ids,
        timestep=timestep,
        target=x0,
    )


def compute_prediction_losses(pred: torch.Tensor, target: torch.Tensor, loss_cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    mse_weight = float(loss_cfg.get("mse_weight", 1.0))
    l1_weight = float(loss_cfg.get("l1_weight", 0.0))

    loss_mse = F.mse_loss(pred, target)
    loss_l1 = F.l1_loss(pred, target)
    loss_total = mse_weight * loss_mse + l1_weight * loss_l1
    return {
        "loss_mse": loss_mse,
        "loss_l1": loss_l1,
        "loss_total": loss_total,
    }