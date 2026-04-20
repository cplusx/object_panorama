from pathlib import Path

import numpy as np
import torch


SAVE_FORMAT_VERSION = 2
LEGACY_MODEL_HIT_WIDTH = 7
LEGACY_MODEL_RGB_OFFSETS = (0, 1, 2)
LEGACY_MODEL_DEPTH_OFFSET = 3
LEGACY_MODEL_NORMAL_OFFSETS = (4, 5, 6)

DEFAULT_MODEL_MAX_HITS = 5
DEFAULT_EDGE_MAX_HITS = 3

MODEL_RGB_STORAGE_DTYPE = "float16"
MODEL_DEPTH_STORAGE_DTYPE = "float16"
MODEL_NORMAL_STORAGE_DTYPE = "float8_e4m3fn_raw_uint8"
EDGE_DEPTH_STORAGE_DTYPE = "float16"
MODEL_NORMAL_FP8_NAME = "float8_e4m3fn"


def _scalar_to_str(value: np.ndarray) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(array.tolist())


def is_mixed_precision_payload(payload: np.lib.npyio.NpzFile) -> bool:
    return "storage_format_version" in payload.files


def split_legacy_model_tensor(model_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_tensor = np.asarray(model_tensor, dtype=np.float32)
    model_rgb = np.stack([model_tensor[offset::LEGACY_MODEL_HIT_WIDTH] for offset in LEGACY_MODEL_RGB_OFFSETS], axis=1)
    model_depth = model_tensor[LEGACY_MODEL_DEPTH_OFFSET::LEGACY_MODEL_HIT_WIDTH]
    model_normal = np.stack([model_tensor[offset::LEGACY_MODEL_HIT_WIDTH] for offset in LEGACY_MODEL_NORMAL_OFFSETS], axis=1)
    return model_rgb.astype(np.float32), model_depth.astype(np.float32), model_normal.astype(np.float32)


def split_legacy_edge_tensor(edge_tensor: np.ndarray, edge_max_hits: int = DEFAULT_EDGE_MAX_HITS) -> np.ndarray:
    return np.asarray(edge_tensor, dtype=np.float32)[: int(edge_max_hits)].astype(np.float32)


def encode_fp8_e4m3fn_to_uint8(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(values.astype(np.float32, copy=False)))
    return tensor.to(torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()


def decode_fp8_e4m3fn_from_uint8(raw_values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(raw_values.astype(np.uint8, copy=False)))
    return tensor.view(torch.float8_e4m3fn).to(torch.float32).cpu().numpy()


def build_mixed_precision_payload(
    *,
    uid: str,
    model_tensor: np.ndarray,
    edge_tensor: np.ndarray,
    resolution: int,
    model_max_hits: int,
    edge_max_hits: int = DEFAULT_EDGE_MAX_HITS,
) -> dict[str, np.ndarray]:
    model_rgb, model_depth, model_normal = split_legacy_model_tensor(model_tensor)
    edge_depth = split_legacy_edge_tensor(edge_tensor, edge_max_hits=edge_max_hits)

    return {
        "uid": np.asarray(uid),
        "storage_format_version": np.asarray(SAVE_FORMAT_VERSION, dtype=np.int32),
        "resolution": np.asarray(int(resolution), dtype=np.int32),
        "width": np.asarray(int(resolution) * 2, dtype=np.int32),
        "model_max_hits": np.asarray(int(model_max_hits), dtype=np.int32),
        "edge_max_hits": np.asarray(int(edge_depth.shape[0]), dtype=np.int32),
        "model_rgb_storage_dtype": np.asarray(MODEL_RGB_STORAGE_DTYPE),
        "model_depth_storage_dtype": np.asarray(MODEL_DEPTH_STORAGE_DTYPE),
        "model_normal_storage_dtype": np.asarray(MODEL_NORMAL_STORAGE_DTYPE),
        "edge_depth_storage_dtype": np.asarray(EDGE_DEPTH_STORAGE_DTYPE),
        "model_normal_fp8_name": np.asarray(MODEL_NORMAL_FP8_NAME),
        "model_rgb": model_rgb.astype(np.float16),
        "model_depth": model_depth.astype(np.float16),
        "model_normal_fp8_e4m3fn_bytes": encode_fp8_e4m3fn_to_uint8(model_normal),
        "edge_depth": edge_depth.astype(np.float16),
    }


def save_mixed_precision_sample(
    path: str | Path,
    *,
    uid: str,
    model_tensor: np.ndarray,
    edge_tensor: np.ndarray,
    resolution: int,
    model_max_hits: int,
    edge_max_hits: int = DEFAULT_EDGE_MAX_HITS,
) -> dict[str, object]:
    path = Path(path)
    payload = build_mixed_precision_payload(
        uid=uid,
        model_tensor=model_tensor,
        edge_tensor=edge_tensor,
        resolution=resolution,
        model_max_hits=model_max_hits,
        edge_max_hits=edge_max_hits,
    )
    temp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp_path, **payload)
    temp_path.replace(path)
    return {
        "storage_format_version": int(payload["storage_format_version"].item()),
        "model_component_shapes": {
            "model_rgb": list(payload["model_rgb"].shape),
            "model_depth": list(payload["model_depth"].shape),
            "model_normal_fp8_e4m3fn_bytes": list(payload["model_normal_fp8_e4m3fn_bytes"].shape),
        },
        "model_rgb_shape": list(payload["model_rgb"].shape),
        "model_depth_shape": list(payload["model_depth"].shape),
        "model_normal_bytes_shape": list(payload["model_normal_fp8_e4m3fn_bytes"].shape),
        "edge_shape": list(payload["edge_depth"].shape),
        "edge_depth_shape": list(payload["edge_depth"].shape),
        "storage_dtypes": {
            "model_rgb": MODEL_RGB_STORAGE_DTYPE,
            "model_depth": MODEL_DEPTH_STORAGE_DTYPE,
            "model_normal": MODEL_NORMAL_STORAGE_DTYPE,
            "edge_depth": EDGE_DEPTH_STORAGE_DTYPE,
        },
        "model_storage_dtypes": {
            "rgb": MODEL_RGB_STORAGE_DTYPE,
            "depth": MODEL_DEPTH_STORAGE_DTYPE,
            "normal": MODEL_NORMAL_STORAGE_DTYPE,
        },
        "edge_storage_dtypes": {
            "depth": EDGE_DEPTH_STORAGE_DTYPE,
        },
    }


def load_sample_modalities(path: str | Path, decode_model_normal: bool = True) -> dict[str, object]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        if is_mixed_precision_payload(payload):
            model_rgb = payload["model_rgb"].astype(np.float32)
            model_depth = payload["model_depth"].astype(np.float32)
            edge_depth = payload["edge_depth"].astype(np.float32)
            raw_normal = payload["model_normal_fp8_e4m3fn_bytes"]
            return {
                "uid": _scalar_to_str(payload["uid"]),
                "storage_format_version": int(np.asarray(payload["storage_format_version"]).item()),
                "resolution": int(np.asarray(payload["resolution"]).item()),
                "width": int(np.asarray(payload["width"]).item()),
                "model_max_hits": int(np.asarray(payload["model_max_hits"]).item()),
                "edge_max_hits": int(np.asarray(payload["edge_max_hits"]).item()),
                "model_rgb": model_rgb,
                "model_depth": model_depth,
                "model_normal": decode_fp8_e4m3fn_from_uint8(raw_normal) if decode_model_normal else None,
                "model_normal_raw": raw_normal,
                "edge_depth": edge_depth,
                "model_component_shapes": {
                    "model_rgb": list(model_rgb.shape),
                    "model_depth": list(model_depth.shape),
                    "model_normal_fp8_e4m3fn_bytes": list(raw_normal.shape),
                },
                "edge_shape": list(edge_depth.shape),
                "storage_dtypes": {
                    "model_rgb": _scalar_to_str(payload["model_rgb_storage_dtype"]),
                    "model_depth": _scalar_to_str(payload["model_depth_storage_dtype"]),
                    "model_normal": _scalar_to_str(payload["model_normal_storage_dtype"]),
                    "edge_depth": _scalar_to_str(payload["edge_depth_storage_dtype"]),
                },
            }

        model_tensor = payload["model_tensor"].astype(np.float32)
        edge_tensor = payload["edge_tensor"].astype(np.float32)
        model_rgb, model_depth, model_normal = split_legacy_model_tensor(model_tensor)
        return {
            "uid": _scalar_to_str(payload["uid"]),
            "storage_format_version": 1,
            "resolution": int(model_tensor.shape[1]),
            "width": int(model_tensor.shape[2]),
            "model_max_hits": int(model_depth.shape[0]),
            "edge_max_hits": int(edge_tensor.shape[0]),
            "model_rgb": model_rgb,
            "model_depth": model_depth,
            "model_normal": model_normal if decode_model_normal else None,
            "model_normal_raw": None,
            "edge_depth": edge_tensor,
            "model_component_shapes": {
                "model_tensor": list(model_tensor.shape),
            },
            "edge_shape": list(edge_tensor.shape),
            "storage_dtypes": {
                "model_tensor": str(model_tensor.dtype),
                "edge_tensor": str(edge_tensor.dtype),
            },
        }