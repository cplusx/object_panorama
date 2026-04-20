from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from evaluation import save_edge3d_validation_preview
from models import RectangularConditionalJiTModel, create_rectangular_conditional_jit_model
from pipeline import Edge3DX0BridgePipeline

from .lr_scheduler_builder import build_lr_scheduler
from .objectives import (
    build_jit_flow_matching_batch,
    compute_prediction_losses,
)
from .optimizer_builder import build_optimizer, freeze_modules_from_config


class RectangularConditionalJiTLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: dict[str, Any],
        objective_cfg: dict[str, Any],
        loss_cfg: dict[str, Any],
        optim_cfg: dict[str, Any],
        freeze_cfg: dict[str, Any] | None,
        pretrained_cfg: dict[str, Any] | None,
        validation_cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.model_cfg = copy.deepcopy(model_cfg)
        self.objective_cfg = copy.deepcopy(objective_cfg)
        self.loss_cfg = copy.deepcopy(loss_cfg)
        self.optim_cfg = copy.deepcopy(optim_cfg)
        self.freeze_cfg = copy.deepcopy(freeze_cfg or {})
        self.pretrained_cfg = copy.deepcopy(pretrained_cfg or {})
        self.validation_cfg = copy.deepcopy(validation_cfg or {})

        effective_model_cfg = _prepare_model_cfg(self.model_cfg, self.objective_cfg)
        if effective_model_cfg.get("preset_name") is None:
            direct_model_cfg = copy.deepcopy(effective_model_cfg)
            direct_model_cfg.pop("preset_name", None)
            self.model = RectangularConditionalJiTModel(**direct_model_cfg)
        else:
            self.model = create_rectangular_conditional_jit_model(**effective_model_cfg)

        load_jit = bool(self.pretrained_cfg.get("load_jit", False))
        checkpoint_path = self.pretrained_cfg.get("public_checkpoint_path")
        if load_jit:
            if not checkpoint_path:
                raise ValueError("pretrained.load_jit=true requires pretrained.public_checkpoint_path")
            self.model.load_pretrained_jit_backbone_from_public_checkpoint(
                checkpoint_path,
                variant=str(self.pretrained_cfg.get("variant", "ema1")),
            )
        freeze_modules_from_config(self.model, self.freeze_cfg)
        self.save_hyperparameters(
            {
                "model_cfg": self.model_cfg,
                "objective_cfg": self.objective_cfg,
                "loss_cfg": self.loss_cfg,
                "optim_cfg": self.optim_cfg,
                "freeze_cfg": self.freeze_cfg,
                "pretrained_cfg": self.pretrained_cfg,
                "validation_cfg": self.validation_cfg,
            }
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
        loss_dict = self._shared_step(batch)
        self._log_losses("train", loss_dict, batch_size=int(batch["edge_depth"].shape[0]))
        return loss_dict["loss_total"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
        pipeline = Edge3DX0BridgePipeline(
            self.model,
            self.objective_cfg,
            inference_dtype=str(self.validation_cfg.get("inference_dtype", "float16")),
        )
        num_steps = int(self.validation_cfg.get("num_inference_steps", 20))
        output = pipeline.generate(batch, num_steps=num_steps, return_intermediates=False)

        _validate_condition_channels(self.model_cfg, int(output["condition_channels"]))
        pred = output["pred_edge_depth"]
        target = torch.nan_to_num(batch["edge_depth"], nan=0.0, posinf=0.0, neginf=0.0).to(
            device=pred.device,
            dtype=pred.dtype,
        )
        infer_loss_dict = compute_prediction_losses(pred, target, self.loss_cfg)
        metrics = {
            "val/infer_loss_total": infer_loss_dict["loss_total"],
            "val/infer_mse": infer_loss_dict["loss_mse"],
            "val/infer_l1": infer_loss_dict["loss_l1"],
        }
        self._log_metric_dict(metrics, batch_size=int(batch["edge_depth"].shape[0]), prog_bar_key="val/infer_loss_total")
        self._maybe_save_validation_preview(batch_idx, batch, pred)
        return metrics

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.optim_cfg)
        scheduler_cfg = copy.deepcopy(self.optim_cfg.get("lr_scheduler"))
        if isinstance(scheduler_cfg, dict):
            scheduler_name = str(scheduler_cfg.get("name", "")).lower()
            if scheduler_name == "cosine_with_warmup" and "total_steps" not in scheduler_cfg and "max_steps" not in scheduler_cfg:
                estimated_steps = self._estimate_total_steps()
                if estimated_steps is not None:
                    scheduler_cfg["total_steps"] = estimated_steps
            elif "max_steps" not in scheduler_cfg and self.optim_cfg.get("max_steps") is not None:
                scheduler_cfg["max_steps"] = int(self.optim_cfg["max_steps"])

        scheduler = build_lr_scheduler(optimizer, scheduler_cfg)
        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _shared_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        model_input = _build_model_input_batch(batch, self.objective_cfg)
        _validate_condition_channels(self.model_cfg, model_input.condition)
        model_output = self.model(
            sample=model_input.sample,
            timestep=model_input.timestep,
            condition=model_input.condition,
            condition_type_ids=model_input.condition_type_ids,
        )
        return compute_prediction_losses(model_output.sample, model_input.target, self.loss_cfg)

    def _log_losses(self, split: str, loss_dict: dict[str, Any], batch_size: int) -> None:
        if getattr(self, "_trainer", None) is None:
            return

        on_step = split == "train"
        sync_dist = bool(getattr(self.trainer, "world_size", 1) > 1)
        self.log(
            f"{split}/loss_total",
            loss_dict["loss_total"],
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        self.log(
            f"{split}/loss_mse",
            loss_dict["loss_mse"],
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )
        self.log(
            f"{split}/loss_l1",
            loss_dict["loss_l1"],
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

    def _log_metric_dict(self, metrics: dict[str, torch.Tensor], batch_size: int, prog_bar_key: str | None = None) -> None:
        if getattr(self, "_trainer", None) is None:
            return

        sync_dist = bool(getattr(self.trainer, "world_size", 1) > 1)
        for name, value in metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == prog_bar_key,
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

    def _maybe_save_validation_preview(self, batch_idx: int, batch: dict[str, Any], pred_edge_depth: torch.Tensor) -> None:
        if batch_idx != 0 or getattr(self, "_trainer", None) is None:
            return

        save_every = int(self.validation_cfg.get("save_preview_every_n_steps", 0))
        if save_every <= 0:
            return

        step = int(getattr(self, "global_step", 0))
        if step % save_every != 0:
            return

        output_dir = Path(self.trainer.default_root_dir) / "validation_previews" / f"step_{step:06d}"
        save_edge3d_validation_preview(output_dir, batch, pred_edge_depth, max_items=2)

    def _estimate_total_steps(self) -> int | None:
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return None
        estimated_steps = getattr(trainer, "estimated_stepping_batches", None)
        if estimated_steps is None:
            return None
        return max(1, int(estimated_steps))


def _build_model_input_batch(batch: dict[str, Any], objective_cfg: dict[str, Any]):
    objective_name = str(objective_cfg.get("name", "flow_matching")).lower()
    if objective_name != "flow_matching":
        raise ValueError(f"Unsupported objective '{objective_name}'")
    return build_jit_flow_matching_batch(
        batch,
        t_min=float(objective_cfg.get("t_min", 0.0)),
        t_max=float(objective_cfg.get("t_max", 1.0)),
        noise_scale=float(objective_cfg.get("noise_scale", 1.0)),
        condition_type_id=int(objective_cfg.get("condition_type_id", 0)),
        use_model_rgb=bool(objective_cfg.get("use_model_rgb", False)),
        use_model_depth=bool(objective_cfg.get("use_model_depth", True)),
        use_model_normal=bool(objective_cfg.get("use_model_normal", True)),
    )


def _prepare_model_cfg(model_cfg: dict[str, Any], objective_cfg: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(model_cfg)
    prepared.pop("name", None)
    condition_channels_per_type = prepared.get("condition_channels_per_type")
    if condition_channels_per_type is not None:
        values = [int(value) for value in condition_channels_per_type]
        if len(values) == 1:
            prepared["condition_channels_per_type"] = [values[0], values[0], values[0]]
        elif len(values) != 3:
            raise ValueError(
                "RectangularConditionalJiT config must provide either 1 or 3 condition channel entries"
            )
    return prepared


def _validate_condition_channels(model_cfg: dict[str, Any], condition: torch.Tensor | int) -> None:
    expected_condition_channels = int(model_cfg["condition_channels_per_type"][0])
    actual_condition_channels = int(condition if isinstance(condition, int) else condition.shape[1])
    if actual_condition_channels != expected_condition_channels:
        raise ValueError(
            f"Condition channel mismatch: model expects {expected_condition_channels}, "
            f"but objective built {actual_condition_channels}"
        )