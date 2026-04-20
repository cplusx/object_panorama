from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataloader_from_config, build_dataset_from_config
from evaluation import save_debug_tensors
from training.objectives import build_jit_flow_matching_batch
from utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a single manifest batch for rectangular conditional JiT training.")
    parser.add_argument("config", help="Path to either a data config or a full experiment config")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    data_cfg, batch_cfg, max_condition_channels, objective_cfg = _resolve_inspection_configs(config, args.split)

    if args.batch_size is not None:
        batch_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        batch_cfg["num_workers"] = args.num_workers

    dataset = build_dataset_from_config(data_cfg)
    dataloader = build_dataloader_from_config(batch_cfg, dataset, max_condition_channels=max_condition_channels)
    batch = next(iter(dataloader))

    inspection_tensors = _resolve_inspection_tensors(batch, objective_cfg=objective_cfg)
    for label, tensor in inspection_tensors["shapes"].items():
        print(f"{label}: {tensor}")

    debug_dir = Path(tempfile.mkdtemp(prefix="rect_cond_jit_manifest_batch_"))
    if inspection_tensors["sample"] is not None:
        save_debug_tensors(
            debug_dir,
            inspection_tensors["sample"],
            inspection_tensors["condition"],
            inspection_tensors["target"],
            inspection_tensors["target"],
        )
    print(f"debug tensors saved to: {debug_dir}")


def _resolve_inspection_configs(config: dict, split: str) -> tuple[dict, dict, int, dict | None]:
    if "data" in config:
        data_cfg = dict(config["data"][split]) if config["data"][split] is not None else None
        if data_cfg is None:
            raise ValueError(f"Experiment config does not define data.{split}")
        train_cfg = dict(config.get("train", {}))
        batch_cfg = {
            "batch_size": int(train_cfg.get("batch_size", 1)),
            "num_workers": int(train_cfg.get("num_workers", 0)),
            "shuffle": False,
            "drop_last": False,
        }
        model_cfg = dict(config.get("model", {}))
        max_condition_channels = max(int(value) for value in model_cfg.get("condition_channels_per_type", [1]))
        return data_cfg, batch_cfg, max_condition_channels, dict(config.get("objective", {}))

    data_cfg = dict(config)
    batch_cfg = {"batch_size": 1, "num_workers": 0, "shuffle": False, "drop_last": False}
    max_condition_channels = 1
    return data_cfg, batch_cfg, max_condition_channels, None


def _resolve_inspection_tensors(batch: dict, objective_cfg: dict | None) -> dict[str, torch.Tensor | dict[str, tuple[int, ...]] | None]:
    if "input" in batch:
        return {
            "sample": batch["input"],
            "condition": batch["condition"],
            "target": batch["target"],
            "shapes": {
                "input shape": tuple(batch["input"].shape),
                "condition shape": tuple(batch["condition"].shape),
                "target shape": tuple(batch["target"].shape),
                "condition_type_ids shape": tuple(batch["condition_type_ids"].shape),
            },
        }

    shapes = {
        "model_rgb shape": tuple(batch["model_rgb"].shape),
        "model_depth shape": tuple(batch["model_depth"].shape),
        "model_normal shape": tuple(batch["model_normal"].shape),
        "edge_depth shape": tuple(batch["edge_depth"].shape),
    }
    if objective_cfg is None:
        return {"sample": None, "condition": None, "target": None, "shapes": shapes}

    model_input = build_jit_flow_matching_batch(
        batch,
        t_min=float(objective_cfg.get("t_min", 0.0)),
        t_max=float(objective_cfg.get("t_max", 1.0)),
        noise_scale=float(objective_cfg.get("noise_scale", 1.0)),
        condition_type_id=int(objective_cfg.get("condition_type_id", 0)),
        use_model_rgb=bool(objective_cfg.get("use_model_rgb", False)),
        use_model_depth=bool(objective_cfg.get("use_model_depth", True)),
        use_model_normal=bool(objective_cfg.get("use_model_normal", True)),
    )
    shapes.update(
        {
            "sample shape": tuple(model_input.sample.shape),
            "condition shape": tuple(model_input.condition.shape),
            "target shape": tuple(model_input.target.shape),
            "condition_type_ids shape": tuple(model_input.condition_type_ids.shape),
        }
    )
    return {
        "sample": model_input.sample,
        "condition": model_input.condition,
        "target": model_input.target,
        "shapes": shapes,
    }


if __name__ == "__main__":
    main()