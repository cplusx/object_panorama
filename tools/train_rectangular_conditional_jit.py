from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataloader_from_config, build_dataset_from_config
from models import create_rectangular_conditional_jit_model
from training import SimpleTrainer, build_lr_scheduler, build_optimizer, freeze_modules_from_config
from utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the rectangular conditional JiT model from an experiment config.")
    parser.add_argument("config", help="Expanded or nested experiment YAML config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_cfg = load_yaml_config(args.config)
    _set_seed(int(experiment_cfg["train"].get("seed", 0)))
    raw_model_cfg = dict(experiment_cfg["model"])
    objective_cfg = dict(experiment_cfg["objective"])
    effective_model_cfg = _prepare_model_cfg(raw_model_cfg)
    model = _build_model_from_config(effective_model_cfg)

    pretrained_cfg = dict(experiment_cfg.get("pretrained", {}))
    load_jit = bool(pretrained_cfg.get("load_jit", False))
    checkpoint_path = pretrained_cfg.get("public_checkpoint_path")
    if load_jit:
        if not checkpoint_path:
            raise ValueError("pretrained.load_jit=true requires pretrained.public_checkpoint_path")
        load_report = model.load_pretrained_jit_backbone_from_public_checkpoint(
            checkpoint_path,
            variant=str(pretrained_cfg.get("variant", "ema1")),
        )
        print(json.dumps(load_report, indent=2))

    freeze_modules_from_config(model, experiment_cfg.get("freeze", {}))
    train_cfg = copy.deepcopy(experiment_cfg["train"])
    optimizer = build_optimizer(model, train_cfg)
    scheduler_cfg = copy.deepcopy(train_cfg.get("lr_scheduler"))
    if isinstance(scheduler_cfg, dict):
        scheduler_cfg["total_steps"] = int(train_cfg["max_steps"])
    scheduler = build_lr_scheduler(optimizer, scheduler_cfg)

    train_dataset = build_dataset_from_config(experiment_cfg["data"]["train"])
    train_loader = build_dataloader_from_config(
        {
            "batch_size": int(train_cfg["batch_size"]),
            "num_workers": int(train_cfg["num_workers"]),
            "shuffle": True,
            "drop_last": False,
            "pin_memory": str(args.device).startswith("cuda"),
        },
        train_dataset,
        max_condition_channels=max(int(value) for value in raw_model_cfg["condition_channels_per_type"]),
    )

    val_loader = None
    if experiment_cfg["data"].get("val") is not None:
        val_dataset = build_dataset_from_config(experiment_cfg["data"]["val"])
        val_loader = build_dataloader_from_config(
            {
                "batch_size": int(train_cfg["batch_size"]),
                "num_workers": int(train_cfg["num_workers"]),
                "shuffle": False,
                "drop_last": False,
                "pin_memory": str(args.device).startswith("cuda"),
            },
            val_dataset,
            max_condition_channels=max(int(value) for value in raw_model_cfg["condition_channels_per_type"]),
        )

    output_dir = Path(experiment_cfg["output_dir"]) / experiment_cfg["experiment_name"]
    trainer = SimpleTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        objective_cfg=objective_cfg,
        loss_cfg=dict(experiment_cfg["loss"]),
        train_cfg=train_cfg,
        device=args.device,
        output_dir=output_dir,
        config_snapshot=experiment_cfg,
    )
    trainer.fit()


def _prepare_model_cfg(model_cfg: dict) -> dict:
    prepared = copy.deepcopy(model_cfg)
    prepared.pop("name", None)
    condition_channels_per_type = prepared.get("condition_channels_per_type")
    if condition_channels_per_type is not None:
        values = [int(value) for value in condition_channels_per_type]
        if len(values) == 1:
            prepared["condition_channels_per_type"] = [values[0], values[0], values[0]]
        elif len(values) != 3:
            raise ValueError("RectangularConditionalJiT config must provide either 1 or 3 condition channel entries")
    return prepared


def _build_model_from_config(model_cfg: dict):
    return create_rectangular_conditional_jit_model(**model_cfg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()