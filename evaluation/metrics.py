from __future__ import annotations

from typing import Iterable

import torch


def loss_dict_to_floats(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().item()) for key, value in loss_dict.items()}


def average_metric_dicts(metric_dicts: Iterable[dict[str, float]]) -> dict[str, float]:
    metrics = list(metric_dicts)
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}