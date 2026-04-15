from __future__ import annotations

from functools import partial
from typing import Any

from torch.utils.data import DataLoader, Dataset

from .collate import conditional_jit_collate_fn
from .manifest_dataset import ConditionalJiTManifestDataset
from .transforms import ComposeDictTransforms, JointRandomHorizontalFlip, JointResize


def build_dataset_from_config(cfg: dict[str, Any]) -> Dataset:
    transforms = []
    resize = cfg.get("resize")
    if resize is not None:
        transforms.append(JointResize(size=(int(resize[0]), int(resize[1]))))
    horizontal_flip_p = float(cfg.get("horizontal_flip_p", 0.0))
    if horizontal_flip_p > 0.0:
        transforms.append(JointRandomHorizontalFlip(horizontal_flip_p))

    dataset_transforms = ComposeDictTransforms(transforms) if transforms else None
    return ConditionalJiTManifestDataset(
        manifest_path=cfg["manifest_path"],
        root_dir=cfg.get("root_dir"),
        transforms=dataset_transforms,
    )


def build_dataloader_from_config(cfg: dict[str, Any], dataset: Dataset, max_condition_channels: int) -> DataLoader:
    batch_size = int(cfg.get("batch_size", 1))
    num_workers = int(cfg.get("num_workers", 0))
    shuffle = bool(cfg.get("shuffle", False))
    drop_last = bool(cfg.get("drop_last", False))
    pin_memory = bool(cfg.get("pin_memory", False))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=partial(conditional_jit_collate_fn, max_condition_channels=max_condition_channels),
    )