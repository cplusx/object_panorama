from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from utils import load_yaml_config


SUPPORTED_ROUNDTRIP_MODES = {"raw", "df"}
DEFAULT_VAE_NAME = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_RAW_SCALE = float(1.0 / math.sqrt(3.0))


def normalize_roundtrip_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_ROUNDTRIP_MODES:
        raise ValueError(f"Unsupported roundtrip mode '{mode}'. Expected one of {sorted(SUPPORTED_ROUNDTRIP_MODES)}")
    return normalized


def default_edge_vae_config(mode: str) -> dict[str, Any]:
    normalized_mode = normalize_roundtrip_mode(mode)
    transform_cfg: dict[str, Any] = {
        "valid_eps": 1.0e-8,
        "decode_valid_threshold": 0.02,
    }
    experiment_name = "edge_raw_sdxl_roundtrip"
    if normalized_mode == "raw":
        transform_cfg["raw_scale"] = DEFAULT_RAW_SCALE
    else:
        experiment_name = "edge_df_sdxl_roundtrip"
        transform_cfg["beta"] = 30.0

    return {
        "experiment_name": experiment_name,
        "vae": {
            "pretrained_model_name_or_path": DEFAULT_VAE_NAME,
            "torch_dtype": "float16",
        },
        "transform": transform_cfg,
        "runtime": {
            "device": "cuda",
        },
    }


def load_edge_vae_config(config_path: str | Path | None, *, mode: str) -> dict[str, Any]:
    config = default_edge_vae_config(mode)
    if config_path is None:
        return config
    loaded = load_yaml_config(config_path)
    return _deep_merge_dicts(config, loaded)


def apply_edge_vae_overrides(
    config: dict[str, Any],
    *,
    vae_name: str | None = None,
    torch_dtype: str | None = None,
    device: str | None = None,
    raw_scale: float | None = None,
    beta: float | None = None,
    valid_eps: float | None = None,
    decode_valid_threshold: float | None = None,
) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    if vae_name is not None:
        merged.setdefault("vae", {})["pretrained_model_name_or_path"] = str(vae_name)
    if torch_dtype is not None:
        merged.setdefault("vae", {})["torch_dtype"] = str(torch_dtype)
    if device is not None:
        merged.setdefault("runtime", {})["device"] = str(device)
    if raw_scale is not None:
        merged.setdefault("transform", {})["raw_scale"] = float(raw_scale)
    if beta is not None:
        merged.setdefault("transform", {})["beta"] = float(beta)
    if valid_eps is not None:
        merged.setdefault("transform", {})["valid_eps"] = float(valid_eps)
    if decode_valid_threshold is not None:
        merged.setdefault("transform", {})["decode_valid_threshold"] = float(decode_valid_threshold)
    return merged


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged