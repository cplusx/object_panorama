from __future__ import annotations

import copy
from typing import Any

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
        max_condition_channels: int,
    ):
        super().__init__()
        self.train_data_cfg = copy.deepcopy(train_data_cfg)
        self.val_data_cfg = copy.deepcopy(val_data_cfg) if val_data_cfg is not None else None
        self.train_cfg = copy.deepcopy(train_cfg)
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
        return build_dataloader_from_config(
            {
                "batch_size": int(self.train_cfg.get("batch_size", 1)),
                "num_workers": int(self.train_cfg.get("num_workers", 0)),
                "shuffle": True,
                "drop_last": bool(self.train_cfg.get("drop_last", False)),
                "pin_memory": bool(self.train_cfg.get("pin_memory", False)),
            },
            self.train_dataset,
            max_condition_channels=self.max_condition_channels,
        )

    def val_dataloader(self):
        if self.val_data_cfg is None:
            return None
        if self.val_dataset is None:
            self.setup(stage="fit")
        return build_dataloader_from_config(
            {
                "batch_size": int(self.train_cfg.get("batch_size", 1)),
                "num_workers": int(self.train_cfg.get("num_workers", 0)),
                "shuffle": False,
                "drop_last": False,
                "pin_memory": bool(self.train_cfg.get("pin_memory", False)),
            },
            self.val_dataset,
            max_condition_channels=self.max_condition_channels,
        )