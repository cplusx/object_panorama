from __future__ import annotations

import torch


def pad_condition_channels(condition: torch.Tensor, target_channels: int) -> torch.Tensor:
    if condition.ndim != 3:
        raise ValueError(f"Expected condition tensor with shape [C, H, W], got {tuple(condition.shape)}")
    if condition.shape[0] > target_channels:
        raise ValueError(
            f"Condition tensor has {condition.shape[0]} channels, which exceeds target_channels={target_channels}"
        )
    if condition.shape[0] == target_channels:
        return condition
    padded = condition.new_zeros((target_channels, condition.shape[1], condition.shape[2]))
    padded[: condition.shape[0]] = condition
    return padded


def conditional_jit_collate_fn(batch: list[dict], max_condition_channels: int) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    reference_input_shape = tuple(batch[0]["input"].shape[-2:])
    reference_condition_shape = tuple(batch[0]["condition"].shape[-2:])
    reference_target_shape = tuple(batch[0]["target"].shape[-2:])

    for item in batch:
        if tuple(item["input"].shape[-2:]) != reference_input_shape:
            raise ValueError("Input spatial sizes do not match within the batch")
        if tuple(item["condition"].shape[-2:]) != reference_condition_shape:
            raise ValueError("Condition spatial sizes do not match within the batch")
        if tuple(item["target"].shape[-2:]) != reference_target_shape:
            raise ValueError("Target spatial sizes do not match within the batch")

    input_tensor = torch.stack([item["input"] for item in batch], dim=0)
    condition_tensor = torch.stack(
        [pad_condition_channels(item["condition"], target_channels=max_condition_channels) for item in batch],
        dim=0,
    )
    target_tensor = torch.stack([item["target"] for item in batch], dim=0)
    condition_type_ids = torch.stack([item["condition_type_id"] for item in batch], dim=0)

    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "input": input_tensor,
        "condition": condition_tensor,
        "target": target_tensor,
        "condition_type_ids": condition_type_ids,
        "meta": [item["meta"] for item in batch],
    }


def edge3d_modality_collate_fn(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    modality_keys = ("model_rgb", "model_depth", "model_normal", "edge_depth")
    reference_shapes = {key: tuple(batch[0][key].shape) for key in modality_keys}

    for item in batch:
        for key in modality_keys:
            if tuple(item[key].shape) != reference_shapes[key]:
                raise ValueError(f"{key} shapes do not match within the batch")

    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "model_rgb": torch.stack([item["model_rgb"] for item in batch], dim=0),
        "model_depth": torch.stack([item["model_depth"] for item in batch], dim=0),
        "model_normal": torch.stack([item["model_normal"] for item in batch], dim=0),
        "edge_depth": torch.stack([item["edge_depth"] for item in batch], dim=0),
        "meta": [item["meta"] for item in batch],
    }