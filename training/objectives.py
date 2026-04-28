from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .types import ModelInputBatch


def build_edge3d_condition(
    batch: dict[str, Any],
    *,
    use_model_rgb: bool = False,
    use_model_depth: bool = True,
    use_model_normal: bool = True,
) -> torch.Tensor:
    parts = []

    if use_model_rgb:
        parts.append(torch.nan_to_num(batch["model_rgb"], nan=0.0, posinf=0.0, neginf=0.0))

    if use_model_depth:
        parts.append(torch.nan_to_num(batch["model_depth"], nan=0.0, posinf=0.0, neginf=0.0))

    if use_model_normal:
        parts.append(torch.nan_to_num(batch["model_normal"], nan=0.0, posinf=0.0, neginf=0.0))

    if not parts:
        raise ValueError("At least one condition modality must be enabled")

    return torch.cat(parts, dim=1)


def build_jit_flow_matching_batch(
    batch: dict[str, Any],
    t_min: float,
    t_max: float,
    noise_scale: float = 1.0,
    condition_dropout_p: float = 0.0,
    condition_type_id: int = 0,
    use_model_rgb: bool = False,
    use_model_depth: bool = True,
    use_model_normal: bool = True,
) -> ModelInputBatch:
    x0 = torch.nan_to_num(batch["edge_depth"], nan=0.0, posinf=0.0, neginf=0.0)
    if x0.shape[1] != 3:
        raise ValueError(f"Expected edge_depth with 3 hits/channels, got {tuple(x0.shape)}")

    noise = torch.randn_like(x0) * float(noise_scale)
    timestep = torch.empty(x0.shape[0], device=x0.device, dtype=x0.dtype).uniform_(float(t_min), float(t_max))
    timestep_view = timestep.view(-1, 1, 1, 1)
    x_t = (1.0 - timestep_view) * x0 + timestep_view * noise

    condition = build_edge3d_condition(
        batch,
        use_model_rgb=use_model_rgb,
        use_model_depth=use_model_depth,
        use_model_normal=use_model_normal,
    )
    if condition_dropout_p > 0.0:
        drop_mask = torch.rand(
            condition.shape[0],
            device=condition.device,
            dtype=condition.dtype,
        ) < float(condition_dropout_p)
        if torch.any(drop_mask):
            condition = condition.clone()
            condition[drop_mask] = 0.0

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
    loss_name = str(loss_cfg.get("name", "weighted_sum")).strip().lower()
    if loss_name in {"weighted_sum", "legacy", "mse_l1"}:
        return _compute_weighted_prediction_losses(pred, target, loss_cfg)
    if loss_name in {"balanced_l2", "balanced_mse"}:
        return _compute_balanced_l2_prediction_losses(pred, target, loss_cfg)
    raise ValueError(f"Unsupported loss '{loss_name}'")


def _compute_weighted_prediction_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
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


def _compute_balanced_l2_prediction_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    edge_threshold = float(loss_cfg.get("edge_threshold", 1.0e-8))

    squared_error = (pred - target).pow(2)
    loss_mse = squared_error.mean()
    loss_l1 = (pred - target).abs().mean()

    edge_pixel_mask = target.amax(dim=1, keepdim=True) > edge_threshold
    non_edge_pixel_mask = ~edge_pixel_mask

    per_sample_metrics = _compute_balanced_region_l2_metrics(
        squared_error,
        edge_pixel_mask=edge_pixel_mask,
        non_edge_pixel_mask=non_edge_pixel_mask,
    )
    loss_total = per_sample_metrics["loss_balanced_l2"].mean()

    return {
        "loss_total": loss_total,
        "loss_balanced_l2": loss_total,
        "loss_mse": loss_mse,
        "loss_l1": loss_l1,
        "loss_edge_l2": per_sample_metrics["loss_edge_l2"].mean(),
        "loss_non_edge_l2": per_sample_metrics["loss_non_edge_l2"].mean(),
        "edge_pixel_fraction": per_sample_metrics["edge_pixel_fraction"].mean(),
        "non_edge_pixel_fraction": per_sample_metrics["non_edge_pixel_fraction"].mean(),
        "edge_weight_scale": per_sample_metrics["edge_weight_scale"].mean(),
        "non_edge_weight_scale": per_sample_metrics["non_edge_weight_scale"].mean(),
    }


def _compute_balanced_region_l2_metrics(
    squared_error: torch.Tensor,
    *,
    edge_pixel_mask: torch.Tensor,
    non_edge_pixel_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    dtype = squared_error.dtype
    device = squared_error.device
    channel_count = int(squared_error.shape[1])

    edge_pixel_count = edge_pixel_mask.sum(dim=(1, 2, 3)).to(dtype=dtype)
    non_edge_pixel_count = non_edge_pixel_mask.sum(dim=(1, 2, 3)).to(dtype=dtype)
    total_pixel_count = edge_pixel_count + non_edge_pixel_count

    edge_error_sum = (squared_error * edge_pixel_mask.to(dtype=dtype)).sum(dim=(1, 2, 3))
    non_edge_error_sum = (squared_error * non_edge_pixel_mask.to(dtype=dtype)).sum(dim=(1, 2, 3))

    edge_element_count = edge_pixel_count * float(channel_count)
    non_edge_element_count = non_edge_pixel_count * float(channel_count)

    edge_l2 = torch.where(
        edge_element_count > 0,
        edge_error_sum / edge_element_count.clamp_min(1.0),
        torch.zeros_like(edge_error_sum, device=device, dtype=dtype),
    )
    non_edge_l2 = torch.where(
        non_edge_element_count > 0,
        non_edge_error_sum / non_edge_element_count.clamp_min(1.0),
        torch.zeros_like(non_edge_error_sum, device=device, dtype=dtype),
    )

    has_edge = edge_pixel_count > 0
    has_non_edge = non_edge_pixel_count > 0
    has_both = has_edge & has_non_edge

    balanced_l2 = torch.where(
        has_both,
        0.5 * (edge_l2 + non_edge_l2),
        torch.where(has_edge, edge_l2, non_edge_l2),
    )

    edge_fraction = edge_pixel_count / total_pixel_count.clamp_min(1.0)
    non_edge_fraction = non_edge_pixel_count / total_pixel_count.clamp_min(1.0)
    edge_weight_scale = torch.where(
        has_both,
        total_pixel_count / (2.0 * edge_pixel_count.clamp_min(1.0)),
        torch.where(has_edge, torch.ones_like(edge_pixel_count), torch.zeros_like(edge_pixel_count)),
    )
    non_edge_weight_scale = torch.where(
        has_both,
        total_pixel_count / (2.0 * non_edge_pixel_count.clamp_min(1.0)),
        torch.where(has_non_edge, torch.ones_like(non_edge_pixel_count), torch.zeros_like(non_edge_pixel_count)),
    )

    return {
        "loss_balanced_l2": balanced_l2,
        "loss_edge_l2": edge_l2,
        "loss_non_edge_l2": non_edge_l2,
        "edge_pixel_fraction": edge_fraction,
        "non_edge_pixel_fraction": non_edge_fraction,
        "edge_weight_scale": edge_weight_scale,
        "non_edge_weight_scale": non_edge_weight_scale,
    }