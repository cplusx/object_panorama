from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from evaluation import average_metric_dicts, loss_dict_to_floats, save_debug_tensors, save_preview_png

from .checkpointing import save_training_checkpoint
from .eval_step import run_eval_step
from .train_step import run_train_step


class SimpleTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        objective_cfg: dict[str, Any],
        loss_cfg: dict[str, Any],
        train_cfg: dict[str, Any],
        device: str | torch.device,
        output_dir: str | Path,
        config_snapshot: dict[str, Any] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.objective_cfg = objective_cfg
        self.loss_cfg = loss_cfg
        self.train_cfg = train_cfg
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.config_snapshot = config_snapshot

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.visual_dir = self.output_dir / "visuals"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.best_val_loss = float("inf")

    def fit(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)
        if self.config_snapshot is not None:
            (self.output_dir / "config_snapshot.yaml").write_text(
                yaml.safe_dump(self.config_snapshot, sort_keys=False),
                encoding="utf-8",
            )

        self.model.to(self.device)
        self.model.train()

        max_steps = int(self.train_cfg["max_steps"])
        log_every = int(self.train_cfg.get("log_every", 0))
        save_every = int(self.train_cfg.get("save_every", 0))
        val_every = int(self.train_cfg.get("val_every", 0))
        visualize_every = int(self.train_cfg.get("visualize_every", 0))
        grad_clip_norm = self.train_cfg.get("grad_clip_norm")

        epoch = 0
        train_iterator = iter(self.train_loader)

        for step in range(1, max_steps + 1):
            try:
                batch = next(train_iterator)
            except StopIteration:
                epoch += 1
                train_iterator = iter(self.train_loader)
                batch = next(train_iterator)

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            step_output = run_train_step(self.model, batch, self.objective_cfg, self.loss_cfg, self.device)
            step_output.loss_total.backward()
            grad_norm_total = self._compute_total_grad_norm()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_clip_norm))

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            train_metrics = self._build_metrics_record(
                step,
                "train",
                step_output.loss_dict,
                grad_norm_total=grad_norm_total,
                pred=step_output.pred,
                target=step_output.target,
            )
            self._append_metrics(train_metrics)

            if log_every > 0 and step % log_every == 0:
                print(json.dumps(train_metrics, ensure_ascii=True))

            if visualize_every > 0 and step % visualize_every == 0:
                visual_step_dir = self.visual_dir / f"step_{step:06d}"
                save_debug_tensors(visual_step_dir, batch, step_output.pred, step_output.target)
                save_preview_png(visual_step_dir, batch, step_output.pred, step_output.target)

            if save_every > 0 and step % save_every == 0:
                self._save_checkpoint(self.checkpoint_dir / "latest.pt", step=step, epoch=epoch)
                self._save_checkpoint(self.checkpoint_dir / f"step_{step:06d}.pt", step=step, epoch=epoch)

            if self.val_loader is not None and val_every > 0 and step % val_every == 0:
                val_metrics = self._run_validation(step)
                if val_metrics["loss_total"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss_total"]
                    self._save_checkpoint(self.checkpoint_dir / "best.pt", step=step, epoch=epoch)

        self._save_checkpoint(self.checkpoint_dir / "latest.pt", step=max_steps, epoch=epoch)

    def _run_validation(self, step: int) -> dict[str, float]:
        self.model.eval()
        metric_dicts = []
        for batch in self.val_loader:
            output = run_eval_step(self.model, batch, self.objective_cfg, self.loss_cfg, self.device)
            metric_dicts.append(loss_dict_to_floats(output.loss_dict))
        averaged = average_metric_dicts(metric_dicts)
        record = {
            "step": int(step),
            "split": "val",
            **averaged,
            "lr_backbone": float(self.optimizer.param_groups[0]["lr"]),
            "lr_new_modules": float(self.optimizer.param_groups[1]["lr"]),
        }
        self._append_metrics(record)
        self.model.train()
        return record

    def _build_metrics_record(
        self,
        step: int,
        split: str,
        loss_dict: dict[str, torch.Tensor],
        grad_norm_total: float | None = None,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
    ) -> dict[str, float | int | str]:
        scalar_losses = loss_dict_to_floats(loss_dict)
        record: dict[str, float | int | str] = {
            "step": int(step),
            "split": split,
            **scalar_losses,
            "lr_backbone": float(self.optimizer.param_groups[0]["lr"]),
            "lr_new_modules": float(self.optimizer.param_groups[1]["lr"]),
        }
        if grad_norm_total is not None:
            record["grad_norm_total"] = float(grad_norm_total)
        if pred is not None:
            record.update(self._tensor_stats(pred, prefix="pred"))
        if target is not None:
            record.update(self._tensor_stats(target, prefix="target"))
        return record

    def _compute_total_grad_norm(self) -> float:
        total_sq_norm = None
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            grad_norm = parameter.grad.detach().float().norm(2)
            if total_sq_norm is None:
                total_sq_norm = grad_norm.pow(2)
            else:
                total_sq_norm = total_sq_norm + grad_norm.pow(2)
        if total_sq_norm is None:
            return 0.0
        return float(total_sq_norm.sqrt().item())

    def _tensor_stats(self, tensor: torch.Tensor, prefix: str) -> dict[str, float]:
        detached = tensor.detach().float()
        return {
            f"{prefix}_min": float(detached.min().item()),
            f"{prefix}_max": float(detached.max().item()),
            f"{prefix}_mean": float(detached.mean().item()),
        }

    def _append_metrics(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _save_checkpoint(self, path: Path, step: int, epoch: int) -> None:
        save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            epoch=epoch,
            extra={
                "best_val_loss": self.best_val_loss,
                "config_snapshot": self.config_snapshot,
            },
        )