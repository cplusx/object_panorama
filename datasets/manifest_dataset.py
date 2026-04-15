from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .tensor_io import load_tensor_file


class ConditionalJiTManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root_dir: str | Path | None = None,
        transforms: callable | None = None,
    ):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        self.root_dir = Path(root_dir).expanduser().resolve() if root_dir is not None else None
        self.transforms = transforms
        self.records = self._load_manifest_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        input_path = self._resolve_data_path(record["input_path"])
        condition_path = self._resolve_data_path(record["condition_path"])
        target_path = self._resolve_data_path(record["target_path"])

        self._assert_file_exists(input_path, record["sample_id"], "input_path")
        self._assert_file_exists(condition_path, record["sample_id"], "condition_path")
        self._assert_file_exists(target_path, record["sample_id"], "target_path")

        sample = {
            "sample_id": record["sample_id"],
            "input": load_tensor_file(input_path),
            "condition": load_tensor_file(condition_path),
            "target": load_tensor_file(target_path),
            "condition_type_id": torch.tensor(record["condition_type_id"], dtype=torch.long),
            "meta": dict(record.get("meta", {})),
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
        required_keys = ["sample_id", "input_path", "condition_path", "target_path", "condition_type_id"]
        for key in required_keys:
            if key not in record:
                raise ValueError(f"Manifest line {line_index} is missing required key '{key}'")
        if int(record["condition_type_id"]) not in {0, 1, 2}:
            raise ValueError(
                f"Manifest line {line_index} has invalid condition_type_id={record['condition_type_id']}; expected 0, 1, or 2"
            )
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