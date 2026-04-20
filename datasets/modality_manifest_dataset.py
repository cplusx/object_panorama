from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from edge3d_tensor_format import load_sample_modalities

from .tensor_io import ensure_chw_float_tensor


class Edge3DModalityManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path | None = None,
        transforms: callable | None = None,
        decode_model_normal: bool = True,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        self.root_dir = Path(root_dir).expanduser().resolve() if root_dir is not None else None
        self.transforms = transforms
        self.decode_model_normal = bool(decode_model_normal)
        self.records = self._load_manifest_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tensor_path = self._resolve_data_path(record["tensor_path"])
        self._assert_file_exists(tensor_path, record["sample_id"], "tensor_path")

        payload = load_sample_modalities(tensor_path, decode_model_normal=self.decode_model_normal)
        if payload["model_normal"] is None:
            raise ValueError(f"Decoded sample is missing model_normal payload: {tensor_path}")

        meta = dict(record.get("meta", {}))
        meta.setdefault("tensor_path", str(tensor_path))
        meta.setdefault("storage_format_version", int(payload["storage_format_version"]))

        sample = {
            "sample_id": record["sample_id"],
            "model_rgb": _flatten_hit_channels(payload["model_rgb"]),
            "model_depth": ensure_chw_float_tensor(payload["model_depth"]),
            "model_normal": _flatten_hit_channels(payload["model_normal"]),
            "edge_depth": ensure_chw_float_tensor(payload["edge_depth"]),
            "meta": meta,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def _load_manifest_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line_index, raw_line in enumerate(self.manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_index} in {self.manifest_path}") from exc
            self._validate_record(record, line_index)
            records.append(record)
        return records

    def _validate_record(self, record: dict[str, Any], line_index: int) -> None:
        required_keys = ["sample_id", "tensor_path"]
        for key in required_keys:
            if key not in record:
                raise ValueError(f"Manifest line {line_index} is missing required key '{key}'")
        if "meta" in record and not isinstance(record["meta"], dict):
            raise ValueError(f"Manifest line {line_index} has non-dict meta field")

    def _resolve_data_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        if self.root_dir is not None:
            return (self.root_dir / path).resolve()
        return (self.manifest_path.parent / path).resolve()

    @staticmethod
    def _assert_file_exists(path: Path, sample_id: str, field_name: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Missing {field_name} for sample '{sample_id}': {path}")


def _flatten_hit_channels(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    if tensor.ndim == 3:
        return ensure_chw_float_tensor(tensor)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a [hits, channels, H, W] tensor, got {tuple(tensor.shape)}")
    hits, channels, height, width = tensor.shape
    return tensor.reshape(hits * channels, height, width).to(dtype=torch.float32)