from __future__ import annotations

from typing import Any

import torch

from .objectives import (
    build_paired_supervised_batch,
    build_x0_prediction_linear_bridge_batch,
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
    )


def _move_batch_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    resolved_device = torch.device(device)
    return {
        "sample_ids": batch["sample_ids"],
        "meta": batch["meta"],
        "input": batch["input"].to(resolved_device),
        "condition": batch["condition"].to(resolved_device),
        "target": batch["target"].to(resolved_device),
        "condition_type_ids": batch["condition_type_ids"].to(resolved_device),
    }


def _build_model_input_batch(batch: dict[str, Any], objective_cfg: dict[str, Any]):
    objective_name = str(objective_cfg["name"]).lower()
    if objective_name == "paired_supervised":
        return build_paired_supervised_batch(batch, fixed_timestep=float(objective_cfg["fixed_timestep"]))
    if objective_name == "x0_prediction_linear_bridge":
        return build_x0_prediction_linear_bridge_batch(
            batch,
            t_min=float(objective_cfg["t_min"]),
            t_max=float(objective_cfg["t_max"]),
            concat_input_to_condition=bool(objective_cfg.get("concat_input_to_condition", False)),
        )
    raise ValueError(f"Unsupported objective '{objective_name}'")