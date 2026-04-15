from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def ensure_chw_float_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    tensor = x if torch.is_tensor(x) else torch.from_numpy(np.asarray(x))
    tensor = tensor.detach().clone() if torch.is_tensor(x) else tensor

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = _maybe_hwc_to_chw(tensor)
    else:
        raise ValueError(f"Expected a 2D or 3D tensor/array, got shape {tuple(tensor.shape)}")
    return tensor.to(dtype=torch.float32)


def load_tensor_file(path: str | Path) -> torch.Tensor:
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {resolved_path}")

    suffix = resolved_path.suffix.lower()
    if suffix == ".pt":
        loaded = torch.load(resolved_path, map_location="cpu")
        if not torch.is_tensor(loaded) and not isinstance(loaded, np.ndarray):
            raise ValueError(f"Expected tensor-like payload in {resolved_path}, got {type(loaded)!r}")
        return ensure_chw_float_tensor(loaded)

    if suffix == ".npy":
        loaded_array = np.load(resolved_path, allow_pickle=False)
        return ensure_chw_float_tensor(loaded_array)

    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        with Image.open(resolved_path) as image:
            if image.mode in {"L", "I;16", "I", "F"}:
                image_array = np.asarray(image)
            else:
                image_array = np.asarray(image.convert("RGB"))
        tensor = ensure_chw_float_tensor(image_array)
        return tensor / 255.0

    raise ValueError(f"Unsupported tensor file format: {resolved_path}")


def _maybe_hwc_to_chw(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] <= 8 and tensor.shape[1] > 8 and tensor.shape[2] > 8:
        return tensor
    if tensor.shape[-1] <= 8 and tensor.shape[0] > 8 and tensor.shape[1] > 8:
        return tensor.permute(2, 0, 1)
    if tensor.shape[-1] in {1, 3, 4} and tensor.shape[0] not in {1, 3, 4}:
        return tensor.permute(2, 0, 1)
    return tensor