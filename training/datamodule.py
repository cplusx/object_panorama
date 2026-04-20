from __future__ import annotations

import copy
from typing import Any

import torch
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from datasets import build_dataloader_from_config, build_dataset_from_config


class RectangularConditionalJiTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_cfg: dict[str, Any],
        val_data_cfg: dict[str, Any] | None,
        train_cfg: dict[str, Any],
        validation_cfg: dict[str, Any] | None,
        max_condition_channels: int,
    ):
        super().__init__()
        self.train_data_cfg = copy.deepcopy(train_data_cfg)
        self.val_data_cfg = copy.deepcopy(val_data_cfg) if val_data_cfg is not None else None
        self.train_cfg = copy.deepcopy(train_cfg)
        self.validation_cfg = copy.deepcopy(validation_cfg or {})
        self.max_condition_channels = int(max_condition_channels)

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = build_dataset_from_config(self.train_data_cfg)
            if self.val_data_cfg is not None and self.val_dataset is None:
                self.val_dataset = build_dataset_from_config(self.val_data_cfg)
        elif stage == "validate" and self.val_data_cfg is not None and self.val_dataset is None:
            self.val_dataset = build_dataset_from_config(self.val_data_cfg)

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup(stage="fit")
        sampler = self._build_train_sampler()
        return build_dataloader_from_config(
            {
                "batch_size": int(self.train_cfg.get("batch_size", 1)),
                "num_workers": int(self.train_cfg.get("num_workers", 0)),
                "shuffle": sampler is None,
                "drop_last": bool(self.train_cfg.get("drop_last", False)),
                "pin_memory": bool(self.train_cfg.get("pin_memory", False)),
                "sampler": sampler,
            },
            self.train_dataset,
            max_condition_channels=self.max_condition_channels,
        )

    def val_dataloader(self):
        if self.val_data_cfg is None:
            return None
        if self.val_dataset is None:
            self.setup(stage="fit")
        local_val_dataset = self._build_local_validation_dataset()
        if local_val_dataset is None:
            return None
        return build_dataloader_from_config(
            {
                "batch_size": int(self.train_cfg.get("batch_size", 1)),
                "num_workers": int(self.train_cfg.get("num_workers", 0)),
                "shuffle": False,
                "drop_last": False,
                "pin_memory": bool(self.train_cfg.get("pin_memory", False)),
            },
            local_val_dataset,
            max_condition_channels=self.max_condition_channels,
        )

    def _build_train_sampler(self):
        rank, world_size = _resolve_rank_and_world_size(getattr(self, "trainer", None))
        if world_size <= 1:
            return None
        return torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=bool(self.train_cfg.get("drop_last", False)),
        )

    def _build_local_validation_dataset(self):
        if self.val_dataset is None:
            return None
        num_val_samples = int(self.validation_cfg.get("num_val_samples", len(self.val_dataset)))
        if num_val_samples <= 0:
            return None
        if len(self.val_dataset) < num_val_samples:
            raise ValueError(
                f"Validation dataset has {len(self.val_dataset)} samples, which is fewer than validation.num_val_samples={num_val_samples}"
            )
        rank, world_size = _resolve_rank_and_world_size(getattr(self, "trainer", None))
        indices = _select_validation_indices(num_val_samples=num_val_samples, rank=rank, world_size=world_size)
        return torch.utils.data.Subset(self.val_dataset, indices)


def _resolve_rank_and_world_size(trainer) -> tuple[int, int]:
    if trainer is None:
        return 0, 1
    rank = int(getattr(trainer, "global_rank", 0))
    world_size = int(getattr(trainer, "world_size", 1))
    return rank, max(1, world_size)


def _select_validation_indices(num_val_samples: int, rank: int, world_size: int) -> list[int]:
    num_val_samples = int(num_val_samples)
    rank = int(rank)
    world_size = max(1, int(world_size))
    if num_val_samples <= 0:
        return []

    base = num_val_samples // world_size
    remainder = num_val_samples % world_size
    local_count = base + (1 if rank < remainder else 0)
    start = rank * base + min(rank, remainder)
    return list(range(start, start + local_count))