from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt


EDGE_DEPTH_NORMALIZATION_SCALE = float(1.0 / math.sqrt(3.0))


def compute_df_void_map(
    valid_mask: np.ndarray,
    *,
    beta: float = 30.0,
) -> np.ndarray:
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected valid_mask with shape [H, W], got {tuple(valid_mask.shape)}")
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")

    height, width = valid_mask.shape
    diag = math.sqrt(float(height * height + width * width))
    if diag <= 0.0:
        raise ValueError("valid_mask must have non-zero spatial extent")

    if np.any(valid_mask):
        distances = distance_transform_edt(~valid_mask).astype(np.float32)
        normalized_distance = np.clip(distances / float(diag), 0.0, 1.0)
        void_df = -np.log2(1.0 + float(beta) * normalized_distance) / np.log2(1.0 + float(beta))
        encoded = np.where(valid_mask, 0.0, void_df)
    else:
        encoded = np.full(valid_mask.shape, fill_value=-1.0, dtype=np.float32)
    return np.asarray(encoded, dtype=np.float32)


def encode_edge_depth_to_df_tensor(
    edge_depth: np.ndarray,
    *,
    beta: float = 30.0,
    valid_eps: float = 1e-8,
) -> np.ndarray:
    edge_depth = _as_edge_depth_tensor(edge_depth)

    encoded = np.empty_like(edge_depth, dtype=np.float32)
    for hit_index in range(edge_depth.shape[0]):
        depth_layer = edge_depth[hit_index]
        valid_mask = _build_valid_mask(depth_layer, valid_eps=valid_eps)
        if not np.any(valid_mask):
            encoded[hit_index] = -1.0
            continue
        normalized_depth = np.clip(depth_layer * EDGE_DEPTH_NORMALIZATION_SCALE, 0.0, 1.0).astype(np.float32)
        void_df = compute_df_void_map(valid_mask, beta=beta)
        encoded[hit_index] = np.where(valid_mask, normalized_depth, void_df)
    return np.clip(encoded, -1.0, 1.0).astype(np.float32)


def decode_df_tensor_to_edge_depth(
    encoded: np.ndarray,
    *,
    valid_threshold: float = 0.02,
) -> np.ndarray:
    encoded = _as_edge_depth_tensor(encoded)
    positive_mask = encoded > float(valid_threshold)
    decoded = np.where(positive_mask, encoded / EDGE_DEPTH_NORMALIZATION_SCALE, 0.0)
    return np.asarray(decoded, dtype=np.float32)


def encode_edge_depth_to_raw_tensor(
    edge_depth: np.ndarray,
    *,
    raw_scale: float = EDGE_DEPTH_NORMALIZATION_SCALE,
    valid_eps: float = 1e-8,
) -> np.ndarray:
    edge_depth = _as_edge_depth_tensor(edge_depth)
    if raw_scale <= 0.0:
        raise ValueError(f"raw_scale must be positive, got {raw_scale}")

    valid_mask = _build_valid_mask(edge_depth, valid_eps=valid_eps)
    encoded = np.where(valid_mask, edge_depth * float(raw_scale), 0.0)
    return np.asarray(encoded, dtype=np.float32)


def decode_raw_tensor_to_edge_depth(
    encoded: np.ndarray,
    *,
    raw_scale: float = EDGE_DEPTH_NORMALIZATION_SCALE,
    valid_threshold: float = 0.02,
) -> np.ndarray:
    encoded = _as_edge_depth_tensor(encoded)
    if raw_scale <= 0.0:
        raise ValueError(f"raw_scale must be positive, got {raw_scale}")

    positive_mask = encoded > float(valid_threshold)
    decoded = np.where(positive_mask, encoded / float(raw_scale), 0.0)
    return np.asarray(decoded, dtype=np.float32)


def _as_edge_depth_tensor(edge_depth: np.ndarray) -> np.ndarray:
    array = np.asarray(edge_depth, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected edge depth tensor with shape [C, H, W], got {tuple(array.shape)}")
    return array.astype(np.float32, copy=False)


def _build_valid_mask(edge_depth: np.ndarray, *, valid_eps: float) -> np.ndarray:
    return np.isfinite(edge_depth) & (edge_depth > float(valid_eps))