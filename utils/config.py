from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    resolved_path = Path(path).expanduser().resolve()
    return _load_yaml_config_internal(resolved_path, visited=set())


def _load_yaml_config_internal(path: Path, visited: set[Path]) -> dict[str, Any]:
    if path in visited:
        chain = " -> ".join(str(item) for item in list(visited) + [path])
        raise ValueError(f"Cyclic YAML config reference detected: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    visited = set(visited)
    visited.add(path)
    raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw_data is None:
        raw_data = {}
    if not isinstance(raw_data, dict):
        raise ValueError(f"Top-level YAML config must be a dict: {path}")
    return _expand_yaml_refs(raw_data, base_dir=path.parent, visited=visited)


def _expand_yaml_refs(value: Any, base_dir: Path, visited: set[Path]) -> Any:
    if isinstance(value, dict):
        return {key: _expand_yaml_refs(item, base_dir=base_dir, visited=visited) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_yaml_refs(item, base_dir=base_dir, visited=visited) for item in value]
    if isinstance(value, str) and value.lower().endswith((".yaml", ".yml")):
        ref_path = Path(value)
        if not ref_path.is_absolute():
            ref_path = _resolve_nested_yaml_path(ref_path, base_dir=base_dir)
        return _load_yaml_config_internal(ref_path, visited=visited)
    return value


def _resolve_nested_yaml_path(ref_path: Path, base_dir: Path) -> Path:
    direct_candidate = (base_dir / ref_path).resolve()
    if direct_candidate.exists():
        return direct_candidate

    for ancestor in [base_dir, *base_dir.parents]:
        candidate = (ancestor / ref_path).resolve()
        if candidate.exists():
            return candidate
    return direct_candidate