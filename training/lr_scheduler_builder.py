from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_lr_scheduler(optimizer: Optimizer, cfg: dict[str, Any] | None):
    if not cfg:
        return None

    scheduler_name = str(cfg.get("name", "")).lower()
    if scheduler_name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if scheduler_name == "cosine_with_warmup":
        warmup_steps = int(cfg.get("warmup_steps", 0))
        total_steps = int(cfg.get("total_steps", cfg.get("max_steps", 0)))
        if total_steps <= 0:
            raise ValueError("cosine_with_warmup scheduler requires total_steps or max_steps > 0")

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported lr scheduler '{scheduler_name}'")