from __future__ import annotations

import copy
from typing import Any

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from models import RectangularConditionalJiTModel, create_rectangular_conditional_jit_model

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
    ):
        super().__init__()
        self.model_cfg = copy.deepcopy(model_cfg)
        self.objective_cfg = copy.deepcopy(objective_cfg)
        self.loss_cfg = copy.deepcopy(loss_cfg)
        self.optim_cfg = copy.deepcopy(optim_cfg)
        self.freeze_cfg = copy.deepcopy(freeze_cfg or {})
        self.pretrained_cfg = copy.deepcopy(pretrained_cfg or {})

        effective_model_cfg = _prepare_model_cfg(self.model_cfg, self.objective_cfg)
        if effective_model_cfg.get("preset_name") is None:
            direct_model_cfg = copy.deepcopy(effective_model_cfg)
            direct_model_cfg.pop("preset_name", None)
            self.model = RectangularConditionalJiTModel(**direct_model_cfg)
        else:
            self.model = create_rectangular_conditional_jit_model(**effective_model_cfg)

        checkpoint_path = self.pretrained_cfg.get("public_checkpoint_path")
        if checkpoint_path:
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
            }
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
        loss_dict = self._shared_step(batch)
        self._log_losses("train", loss_dict, batch_size=int(batch["edge_depth"].shape[0]))
        return loss_dict["loss_total"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Any:
        loss_dict = self._shared_step(batch)
        self._log_losses("val", loss_dict, batch_size=int(batch["edge_depth"].shape[0]))
        return loss_dict["loss_total"]

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
    )


def _prepare_model_cfg(model_cfg: dict[str, Any], objective_cfg: dict[str, Any]) -> dict[str, Any]:
    prepared = copy.deepcopy(model_cfg)
    prepared.pop("name", None)
    return prepared