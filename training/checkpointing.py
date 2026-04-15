from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    extra: dict[str, Any],
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "extra": dict(extra),
        "config_snapshot": dict(extra.get("config_snapshot", {})) if isinstance(extra.get("config_snapshot"), dict) else None,
    }
    torch.save(payload, checkpoint_path)


def load_training_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return {
        "step": int(checkpoint.get("step", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "extra": dict(checkpoint.get("extra", {})),
    }