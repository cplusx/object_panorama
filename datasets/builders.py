from __future__ import annotations

import copy
import json
from functools import partial
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset

from .collate import conditional_jit_collate_fn, edge3d_modality_collate_fn
from .manifest_dataset import ConditionalJiTManifestDataset
from .modality_manifest_dataset import Edge3DModalityManifestDataset
from .transforms import ComposeDictTransforms, JointRandomHorizontalFlip, JointResize


def build_dataset_from_config(cfg: dict[str, Any]) -> Dataset:
    cfg = copy.deepcopy(cfg)
    transforms = []
    resize = cfg.get("resize")
    if resize is not None:
        transforms.append(JointResize(size=(int(resize[0]), int(resize[1]))))
    horizontal_flip_p = float(cfg.get("horizontal_flip_p", 0.0))
    if horizontal_flip_p > 0.0:
        transforms.append(JointRandomHorizontalFlip(horizontal_flip_p))

    dataset_transforms = ComposeDictTransforms(transforms) if transforms else None
    dataset_type = str(cfg.get("dataset_type", "conditional_jit_manifest")).lower()
    if dataset_type in {"edge3d_modalities", "edge3d_modality_manifest", "modality_manifest"}:
        cfg = resolve_or_create_edge3d_manifest_from_config(cfg)
        return Edge3DModalityManifestDataset(
            manifest_path=cfg["manifest_path"],
            root_dir=cfg.get("root_dir"),
            transforms=dataset_transforms,
            decode_model_normal=bool(cfg.get("decode_model_normal", True)),
        )
    return ConditionalJiTManifestDataset(
        manifest_path=cfg["manifest_path"],
        root_dir=cfg.get("root_dir"),
        transforms=dataset_transforms,
    )


def resolve_or_create_edge3d_manifest_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    resolved_cfg = copy.deepcopy(cfg)
    manifest_path = Path(resolved_cfg["manifest_path"]).expanduser().resolve()
    overwrite_manifest = bool(resolved_cfg.get("overwrite_manifest", False))

    if manifest_path.exists() and not overwrite_manifest:
        resolved_cfg["manifest_path"] = str(manifest_path)
        resolved_cfg.pop("root_dir", None)
        return resolved_cfg

    data_folders = _normalize_data_folders(resolved_cfg.get("data_folders"))
    selected_paths = _select_edge3d_npz_paths(
        data_folders=data_folders,
        start_index=int(resolved_cfg.get("start_index", 0)),
        data_number=resolved_cfg.get("data_number"),
    )
    if not selected_paths:
        raise ValueError("Edge3D manifest selection produced zero samples")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for tensor_path in selected_paths:
            record = {
                "sample_id": tensor_path.stem,
                "tensor_path": str(tensor_path),
                "meta": {"source_folder": str(tensor_path.parent)},
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    resolved_cfg["manifest_path"] = str(manifest_path)
    resolved_cfg.pop("root_dir", None)
    return resolved_cfg


def _normalize_data_folders(value: Any) -> list[Path]:
    if isinstance(value, (str, Path)):
        raw_folders = [value]
    elif isinstance(value, list):
        raw_folders = value
    else:
        raise ValueError("Edge3D modality configs must define data_folders as a string or list")

    folders = [Path(folder).expanduser().resolve() for folder in raw_folders]
    if not folders:
        raise ValueError("Edge3D modality configs must define at least one data folder")
    for folder in folders:
        if not folder.exists():
            raise FileNotFoundError(f"Edge3D data folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Edge3D data folder is not a directory: {folder}")
    return folders


def _select_edge3d_npz_paths(
    data_folders: list[Path],
    start_index: int,
    data_number: Any,
) -> list[Path]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    tensor_paths: list[Path] = []
    for folder in data_folders:
        tensor_paths.extend(path.resolve() for path in folder.glob("*.npz"))

    tensor_paths = sorted(tensor_paths, key=lambda path: (path.name, str(path)))
    if data_number is None:
        return tensor_paths[start_index:]

    count = int(data_number)
    if count < 0:
        raise ValueError("data_number must be non-negative or null")
    return tensor_paths[start_index : start_index + count]


def build_dataloader_from_config(cfg: dict[str, Any], dataset: Dataset, max_condition_channels: int) -> DataLoader:
    batch_size = int(cfg.get("batch_size", 1))
    num_workers = int(cfg.get("num_workers", 0))
    shuffle = bool(cfg.get("shuffle", False))
    drop_last = bool(cfg.get("drop_last", False))
    pin_memory = bool(cfg.get("pin_memory", False))

    collate_fn = edge3d_modality_collate_fn if _is_edge3d_modality_dataset(dataset) else partial(
        conditional_jit_collate_fn,
        max_condition_channels=max_condition_channels,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


def _is_edge3d_modality_dataset(dataset: Dataset) -> bool:
    current = dataset
    visited: set[int] = set()
    while True:
        if isinstance(current, Edge3DModalityManifestDataset):
            return True
        next_dataset = getattr(current, "dataset", None)
        if next_dataset is None:
            return False
        if id(next_dataset) in visited:
            return False
        visited.add(id(next_dataset))
        current = next_dataset