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
        rank, world_size = _resolve_rank_and_world_size(_get_attached_trainer(self))
        micro_batches_per_epoch = _resolve_train_micro_batches_per_epoch(self.train_cfg, world_size)
        if micro_batches_per_epoch is None:
            if world_size <= 1:
                return None
            return torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=bool(self.train_cfg.get("drop_last", False)),
            )

        batch_size = int(self.train_cfg.get("batch_size", 1))
        return _RepeatingDistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(self.train_cfg.get("seed", 0)),
            num_samples=batch_size * micro_batches_per_epoch,
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
        rank, world_size = _resolve_rank_and_world_size(_get_attached_trainer(self))
        indices = _select_validation_indices(num_val_samples=num_val_samples, rank=rank, world_size=world_size)
        return torch.utils.data.Subset(self.val_dataset, indices)


def _resolve_rank_and_world_size(trainer) -> tuple[int, int]:
    if trainer is None:
        return 0, 1
    rank = int(getattr(trainer, "global_rank", 0))
    world_size = int(getattr(trainer, "world_size", 1))
    return rank, max(1, world_size)


def _get_attached_trainer(datamodule) -> Any:
    trainer = getattr(datamodule, "_trainer", None)
    if trainer is not None:
        return trainer
    return getattr(datamodule, "trainer", None)


class _RepeatingDistributedSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        dataset,
        *,
        num_samples: int,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if len(dataset) <= 0:
            raise ValueError("Training dataset must contain at least one sample")
        self.dataset = dataset
        self.num_samples = int(num_samples)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        dataset_size = len(self.dataset)
        indices: list[int] = []
        while len(indices) < self.total_size:
            if self.shuffle:
                indices.extend(torch.randperm(dataset_size, generator=generator).tolist())
            else:
                indices.extend(range(dataset_size))
        indices = indices[: self.total_size]
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _resolve_train_micro_batches_per_epoch(train_cfg: dict[str, Any], world_size: int) -> int | None:
    effective_steps_per_epoch = train_cfg.get("effective_steps_per_epoch")
    effective_batch_size = train_cfg.get("effective_batch_size")
    if effective_steps_per_epoch is None or effective_batch_size is None:
        return None

    batch_size = int(train_cfg.get("batch_size", 1))
    global_micro_batch = batch_size * max(1, int(world_size))
    total_batch = int(effective_batch_size)
    if total_batch % global_micro_batch != 0:
        raise ValueError(
            f"effective_batch_size={total_batch} must be divisible by batch_size*world_size={global_micro_batch}"
        )

    accumulate_grad_batches = total_batch // global_micro_batch
    return int(effective_steps_per_epoch) * accumulate_grad_batches


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