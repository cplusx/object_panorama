from __future__ import annotations

from typing import Any

import torch

from .train_step import run_train_step
from .types import TrainStepOutput


def run_eval_step(
    model,
    batch: dict[str, Any],
    objective_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    device: str | torch.device,
) -> TrainStepOutput:
    with torch.no_grad():
        return run_train_step(model, batch, objective_cfg, loss_cfg, device)