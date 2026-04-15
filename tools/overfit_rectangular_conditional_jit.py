from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataloader_from_config, build_dataset_from_config
from models import create_rectangular_conditional_jit_model
from training import SimpleTrainer, build_lr_scheduler, build_optimizer, freeze_modules_from_config
from utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a small subset with the rectangular conditional JiT trainer.")
    parser.add_argument("config", help="Expanded or nested experiment YAML config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_cfg = load_yaml_config(args.config)
    _set_seed(int(experiment_cfg["train"].get("seed", 0)))
    raw_model_cfg = dict(experiment_cfg["model"])
    effective_model_cfg = _prepare_model_cfg(raw_model_cfg, objective_cfg={"name": "paired_supervised"})
    model = _build_model_from_config(effective_model_cfg)

    train_dataset = build_dataset_from_config(experiment_cfg["data"]["train"])
    subset_size = min(int(args.num_samples), len(train_dataset))
    if subset_size <= 0:
        raise ValueError("Overfit dataset is empty")
    train_dataset = Subset(train_dataset, list(range(subset_size)))

    train_cfg = copy.deepcopy(experiment_cfg["train"])
    train_cfg["max_steps"] = int(args.max_steps)
    train_cfg["log_every"] = 20
    train_cfg["save_every"] = 50
    train_cfg["visualize_every"] = 50
    train_cfg["val_every"] = 0

    freeze_modules_from_config(model, experiment_cfg.get("freeze", {}))
    optimizer = build_optimizer(model, train_cfg)
    scheduler_cfg = copy.deepcopy(train_cfg.get("lr_scheduler"))
    if isinstance(scheduler_cfg, dict):
        scheduler_cfg["total_steps"] = int(train_cfg["max_steps"])
    scheduler = build_lr_scheduler(optimizer, scheduler_cfg)

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

    objective_cfg = {
        "name": "paired_supervised",
        "fixed_timestep": float(experiment_cfg.get("objective", {}).get("fixed_timestep", 0.5)),
    }
    output_dir = Path(experiment_cfg["output_dir"]) / experiment_cfg["experiment_name"]
    trainer = SimpleTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=None,
        objective_cfg=objective_cfg,
        loss_cfg=dict(experiment_cfg["loss"]),
        train_cfg=train_cfg,
        device=args.device,
        output_dir=output_dir,
        config_snapshot=experiment_cfg,
    )
    trainer.fit()


def _prepare_model_cfg(model_cfg: dict, objective_cfg: dict) -> dict:
    prepared = copy.deepcopy(model_cfg)
    prepared.pop("name", None)
    if objective_cfg.get("name") == "x0_prediction_linear_bridge" and objective_cfg.get("concat_input_to_condition", False):
        image_in_channels = int(prepared["image_in_channels"])
        prepared["condition_channels_per_type"] = [
            int(value) + image_in_channels for value in prepared["condition_channels_per_type"]
        ]
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