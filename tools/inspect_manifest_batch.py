from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataloader_from_config, build_dataset_from_config
from evaluation import save_debug_tensors
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
    data_cfg, batch_cfg, max_condition_channels = _resolve_inspection_configs(config, args.split)

    if args.batch_size is not None:
        batch_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        batch_cfg["num_workers"] = args.num_workers

    dataset = build_dataset_from_config(data_cfg)
    dataloader = build_dataloader_from_config(batch_cfg, dataset, max_condition_channels=max_condition_channels)
    batch = next(iter(dataloader))

    print(f"input shape: {tuple(batch['input'].shape)}")
    print(f"condition shape: {tuple(batch['condition'].shape)}")
    print(f"target shape: {tuple(batch['target'].shape)}")
    print(f"condition_type_ids shape: {tuple(batch['condition_type_ids'].shape)}")

    debug_dir = Path(tempfile.mkdtemp(prefix="rect_cond_jit_manifest_batch_"))
    save_debug_tensors(debug_dir, batch, pred=batch["target"], target=batch["target"])
    print(f"debug tensors saved to: {debug_dir}")


def _resolve_inspection_configs(config: dict, split: str) -> tuple[dict, dict, int]:
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
        return data_cfg, batch_cfg, max_condition_channels

    data_cfg = dict(config)
    batch_cfg = {"batch_size": 1, "num_workers": 0, "shuffle": False, "drop_last": False}
    dataset = build_dataset_from_config(data_cfg)
    max_condition_channels = max(int(dataset[index]["condition"].shape[0]) for index in range(len(dataset)))
    return data_cfg, batch_cfg, max_condition_channels


if __name__ == "__main__":
    main()