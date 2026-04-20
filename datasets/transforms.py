from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


_TRANSFORM_TENSOR_KEYS = (
    "input",
    "condition",
    "target",
    "model_rgb",
    "model_depth",
    "model_normal",
    "edge_depth",
)


class JointResize:
    def __init__(self, size: tuple[int, int], mode: str = "bilinear"):
        self.size = (int(size[0]), int(size[1]))
        self.mode = str(mode)

    def __call__(self, sample: dict) -> dict:
        output = dict(sample)
        for key in _TRANSFORM_TENSOR_KEYS:
            if key in output:
                output[key] = _resize_tensor(output[key], size=self.size, mode=self.mode)
        return output


class JointRandomHorizontalFlip:
    def __init__(self, p: float):
        self.p = float(p)

    def __call__(self, sample: dict) -> dict:
        if self.p <= 0.0 or torch.rand(1).item() >= self.p:
            return sample
        output = dict(sample)
        for key in _TRANSFORM_TENSOR_KEYS:
            if key in output:
                output[key] = torch.flip(output[key], dims=[-1])
        return output


class ComposeDictTransforms:
    def __init__(self, transforms: list[Callable[[dict], dict]]):
        self.transforms = list(transforms)

    def __call__(self, sample: dict) -> dict:
        output = sample
        for transform in self.transforms:
            output = transform(output)
        return output


def _resize_tensor(tensor: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected CHW tensor for resizing, got shape {tuple(tensor.shape)}")
    batch_tensor = tensor.unsqueeze(0)
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        resized = F.interpolate(batch_tensor, size=size, mode=mode, align_corners=False)
    else:
        resized = F.interpolate(batch_tensor, size=size, mode=mode)
    return resized.squeeze(0)