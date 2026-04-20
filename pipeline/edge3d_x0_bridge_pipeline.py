from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any

import torch

from training.objectives import build_edge3d_condition


class Edge3DX0BridgePipeline:
    def __init__(self, model, objective_cfg: dict[str, Any], inference_dtype: str = "float16"):
        self.model = model
        self.objective_cfg = copy.deepcopy(objective_cfg)
        self.inference_dtype = _normalize_inference_dtype(inference_dtype)

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_batch(self, batch: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        device = self._model_device()
        edge_depth = torch.nan_to_num(batch["edge_depth"], nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.is_floating_point(edge_depth):
            edge_depth = edge_depth.to(dtype=torch.float32)
        dtype = edge_depth.dtype

        device_batch = {
            "sample_ids": batch.get("sample_ids", []),
            "meta": batch.get("meta", []),
            "model_rgb": torch.nan_to_num(batch["model_rgb"], nan=0.0, posinf=0.0, neginf=0.0).to(device=device, dtype=dtype),
            "model_depth": torch.nan_to_num(batch["model_depth"], nan=0.0, posinf=0.0, neginf=0.0).to(device=device, dtype=dtype),
            "model_normal": torch.nan_to_num(batch["model_normal"], nan=0.0, posinf=0.0, neginf=0.0).to(device=device, dtype=dtype),
            "edge_depth": edge_depth.to(device=device, dtype=dtype),
        }

        condition = build_edge3d_condition(
            device_batch,
            use_model_rgb=bool(self.objective_cfg.get("use_model_rgb", False)),
            use_model_depth=bool(self.objective_cfg.get("use_model_depth", True)),
            use_model_normal=bool(self.objective_cfg.get("use_model_normal", True)),
        )
        self._validate_condition_channels(condition)

        condition_type_ids = torch.full(
            (device_batch["edge_depth"].shape[0],),
            int(self.objective_cfg.get("condition_type_id", 0)),
            device=device,
            dtype=torch.long,
        )
        return device_batch, condition, condition_type_ids

    def _validate_condition_channels(self, condition: torch.Tensor) -> None:
        expected_values = getattr(getattr(self.model, "config", None), "condition_channels_per_type", None)
        if expected_values is None:
            return
        expected_condition_channels = int(expected_values[0])
        actual_condition_channels = int(condition.shape[1])
        if actual_condition_channels != expected_condition_channels:
            raise ValueError(
                f"Condition channel mismatch: model expects {expected_condition_channels}, "
                f"but objective built {actual_condition_channels}"
            )

    @torch.no_grad()
    def generate(
        self,
        batch: dict[str, Any],
        num_steps: int = 20,
        noise: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, Any]:
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        device_batch, condition, condition_type_ids = self._prepare_batch(batch)
        target = device_batch["edge_depth"]
        device = target.device
        dtype = target.dtype
        x0_shape = tuple(target.shape)

        if noise is None:
            z = torch.randn_like(target)
        else:
            if tuple(noise.shape) != x0_shape:
                raise ValueError(f"Expected noise shape {x0_shape}, got {tuple(noise.shape)}")
            z = noise.to(device=device, dtype=dtype)

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
        x_t = z
        pred_x0 = z
        intermediates: list[dict[str, torch.Tensor | float]] | None = [] if return_intermediates else None

        was_training = bool(self.model.training)
        self.model.eval()
        use_autocast = device.type == "cuda" and self.inference_dtype == "float16"
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_autocast else nullcontext()
        effective_dtype = "float16" if use_autocast else "float32"

        try:
            with autocast_context:
                for index in range(num_steps):
                    t = timesteps[index]
                    t_next = timesteps[index + 1]
                    timestep_batch = torch.full((x0_shape[0],), float(t.item()), device=device, dtype=dtype)
                    pred_x0 = self.model(
                        sample=x_t,
                        timestep=timestep_batch,
                        condition=condition,
                        condition_type_ids=condition_type_ids,
                    ).sample
                    x_t = (1.0 - t_next) * pred_x0 + t_next * z

                    if intermediates is not None:
                        intermediates.append(
                            {
                                "timestep": float(t.item()),
                                "pred_x0": pred_x0.detach().to(dtype=torch.float32).cpu().clone(),
                                "x_t": x_t.detach().to(dtype=torch.float32).cpu().clone(),
                            }
                        )
        finally:
            self.model.train(was_training)

        output = {
            "pred_edge_depth": pred_x0.detach().to(dtype=torch.float32),
            "initial_noise": z.detach().to(dtype=torch.float32),
            "num_steps": num_steps,
            "condition_channels": int(condition.shape[1]),
            "effective_inference_dtype": effective_dtype,
        }
        if intermediates is not None:
            output["intermediates"] = intermediates
        return output


def _normalize_inference_dtype(inference_dtype: str) -> str:
    normalized = str(inference_dtype).strip().lower()
    if normalized not in {"float16", "float32"}:
        raise ValueError("inference_dtype must be 'float16' or 'float32'")
    return normalized