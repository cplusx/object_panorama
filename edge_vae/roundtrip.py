from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from edge3d.tensor_format import load_sample_modalities

from .codec import DiffusersVAECodec
from .configs import normalize_roundtrip_mode
from .transforms import (
    decode_df_tensor_to_edge_depth,
    decode_raw_tensor_to_edge_depth,
    encode_edge_depth_to_df_tensor,
    encode_edge_depth_to_raw_tensor,
)
from .visualization import save_edge_vae_outputs


def run_edge_vae_roundtrip(
    sample_path: str,
    output_dir: str,
    *,
    mode: str,
    vae_cfg: dict,
    transform_cfg: dict,
    runtime_cfg: dict,
):
    normalized_mode = normalize_roundtrip_mode(mode)
    payload = load_sample_modalities(sample_path, decode_model_normal=False)
    target_edge_depth = np.asarray(payload["edge_depth"], dtype=np.float32)

    input_encoded = _encode_edge_depth(target_edge_depth, mode=normalized_mode, transform_cfg=transform_cfg)
    codec = DiffusersVAECodec(
        pretrained_model_name_or_path=str(vae_cfg.get("pretrained_model_name_or_path", "madebyollin/sdxl-vae-fp16-fix")),
        torch_dtype=str(vae_cfg.get("torch_dtype", "float16")),
        device=str(runtime_cfg.get("device", "cuda")),
    )

    roundtrip_output = codec.roundtrip(torch.from_numpy(input_encoded[None, ...]))
    decoded_encoded = roundtrip_output["decoded"].detach().to(dtype=torch.float32).cpu().numpy()[0]
    decoded_edge_depth = _decode_edge_depth(decoded_encoded, mode=normalized_mode, transform_cfg=transform_cfg)

    artifacts = save_edge_vae_outputs(
        output_dir,
        mode=normalized_mode,
        target_edge_depth=target_edge_depth,
        input_encoded=input_encoded,
        decoded_encoded=decoded_encoded,
        decoded_edge_depth=decoded_edge_depth,
        decode_valid_threshold=float(transform_cfg.get("decode_valid_threshold", 0.02)),
    )

    return {
        "sample_id": str(payload.get("uid", Path(sample_path).stem)),
        "mode": normalized_mode,
        "output_dir": str(Path(output_dir).resolve()),
        "resolved_device": str(codec.device),
        "resolved_torch_dtype": str(codec.torch_dtype).replace("torch.", ""),
        "artifacts": artifacts,
    }


def _encode_edge_depth(edge_depth: np.ndarray, *, mode: str, transform_cfg: dict[str, Any]) -> np.ndarray:
    if mode == "raw":
        return encode_edge_depth_to_raw_tensor(
            edge_depth,
            depth_scale=float(transform_cfg.get("depth_scale", 2.0)),
            raw_scale=float(transform_cfg.get("raw_scale", 1.0 / np.sqrt(3.0))),
            valid_eps=float(transform_cfg.get("valid_eps", 1.0e-8)),
        )
    return encode_edge_depth_to_df_tensor(
        edge_depth,
        beta=float(transform_cfg.get("beta", 30.0)),
        depth_scale=float(transform_cfg.get("depth_scale", 2.0)),
        valid_eps=float(transform_cfg.get("valid_eps", 1.0e-8)),
    )


def _decode_edge_depth(encoded: np.ndarray, *, mode: str, transform_cfg: dict[str, Any]) -> np.ndarray:
    if mode == "raw":
        return decode_raw_tensor_to_edge_depth(
            encoded,
            depth_scale=float(transform_cfg.get("depth_scale", 2.0)),
            raw_scale=float(transform_cfg.get("raw_scale", 1.0 / np.sqrt(3.0))),
            valid_threshold=float(transform_cfg.get("decode_valid_threshold", 0.02)),
        )
    return decode_df_tensor_to_edge_depth(
        encoded,
        depth_scale=float(transform_cfg.get("depth_scale", 2.0)),
        valid_threshold=float(transform_cfg.get("decode_valid_threshold", 0.02)),
    )