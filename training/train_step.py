from __future__ import annotations

from typing import Any

import torch

from .objectives import (
    build_jit_flow_matching_batch,
    compute_prediction_losses,
)
from .types import TrainStepOutput


def run_train_step(
    model,
    batch: dict[str, Any],
    objective_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    device: str | torch.device,
) -> TrainStepOutput:
    device_batch = _move_batch_to_device(batch, device)
    model_input = _build_model_input_batch(device_batch, objective_cfg)
    model_output = model(
        sample=model_input.sample,
        timestep=model_input.timestep,
        condition=model_input.condition,
        condition_type_ids=model_input.condition_type_ids,
    )
    loss_dict = compute_prediction_losses(model_output.sample, model_input.target, loss_cfg)
    return TrainStepOutput(
        loss_total=loss_dict["loss_total"],
        loss_dict=loss_dict,
        pred=model_output.sample,
        target=model_input.target,
        sample=model_input.sample,
        condition=model_input.condition,
    )


def _move_batch_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    resolved_device = torch.device(device)
    return {
        "sample_ids": batch["sample_ids"],
        "meta": batch["meta"],
        "model_rgb": batch["model_rgb"].to(resolved_device),
        "model_depth": batch["model_depth"].to(resolved_device),
        "model_normal": batch["model_normal"].to(resolved_device),
        "edge_depth": batch["edge_depth"].to(resolved_device),
    }


def _build_model_input_batch(batch: dict[str, Any], objective_cfg: dict[str, Any]):
    objective_name = str(objective_cfg.get("name", "flow_matching")).lower()
    if objective_name != "flow_matching":
        raise ValueError(f"Unsupported objective '{objective_name}'")
    return build_jit_flow_matching_batch(
        batch,
        t_min=float(objective_cfg.get("t_min", 0.0)),
        t_max=float(objective_cfg.get("t_max", 1.0)),
        noise_scale=float(objective_cfg.get("noise_scale", 1.0)),
        condition_dropout_p=float(objective_cfg.get("condition_dropout_p", 0.0)),
        condition_type_id=int(objective_cfg.get("condition_type_id", 0)),
        use_model_rgb=bool(objective_cfg.get("use_model_rgb", False)),
        use_model_depth=bool(objective_cfg.get("use_model_depth", True)),
        use_model_normal=bool(objective_cfg.get("use_model_normal", True)),
    )