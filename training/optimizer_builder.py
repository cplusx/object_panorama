from __future__ import annotations

from typing import Any

import torch


BACKBONE_PREFIXES = ("t_embedder.", "x_embedder.", "blocks.", "final_layer.")
NEW_MODULE_PREFIXES = (
    "image_input_adapter.",
    "image_output_adapter.",
    "condition_stems.",
    "condition_tower.",
    "g_proj.",
    "interaction_blocks.",
)


def build_param_groups(model: torch.nn.Module, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base_lr = float(cfg["lr"])
    backbone_lr_mult = float(cfg.get("backbone_lr_mult", 1.0))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    backbone_params = []
    new_module_params = []
    unassigned_names = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(BACKBONE_PREFIXES):
            backbone_params.append(parameter)
        elif name.startswith(NEW_MODULE_PREFIXES):
            new_module_params.append(parameter)
        else:
            unassigned_names.append(name)

    if unassigned_names:
        raise ValueError(f"Unassigned trainable parameters found: {unassigned_names}")

    return [
        {
            "name": "backbone",
            "params": backbone_params,
            "lr": base_lr * backbone_lr_mult,
            "weight_decay": weight_decay,
        },
        {
            "name": "new_modules",
            "params": new_module_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
        },
    ]


def freeze_modules_from_config(model: torch.nn.Module, cfg: dict[str, Any]) -> None:
    if cfg.get("freeze_t_embedder", False):
        _set_requires_grad(model.t_embedder, requires_grad=False)
    if cfg.get("freeze_x_embedder", False):
        _set_requires_grad(model.x_embedder, requires_grad=False)
    if cfg.get("freeze_final_layer", False):
        _set_requires_grad(model.final_layer, requires_grad=False)

    freeze_blocks_before = cfg.get("freeze_blocks_before")
    if freeze_blocks_before is not None:
        block_limit = int(freeze_blocks_before)
        for block in model.blocks[:block_limit]:
            _set_requires_grad(block, requires_grad=False)


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(cfg.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'")
    param_groups = build_param_groups(model, cfg)
    return torch.optim.AdamW(param_groups)


def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad