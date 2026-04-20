from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

from training.datamodule import RectangularConditionalJiTDataModule
from training.lightning_module import RectangularConditionalJiTLightningModule
from utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal training entrypoint for the rectangular conditional JiT model via PyTorch Lightning.")
    parser.add_argument("config", help="Expanded or nested experiment YAML config")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument(
        "--strategy",
        choices=["auto", "ddp", "deepspeed_stage_2", "deepspeed_stage_2_offload"],
        default=None,
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--limit-train-batches", default=None)
    parser.add_argument("--limit-val-batches", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_cfg = load_yaml_config(args.config)
    train_cfg = dict(experiment_cfg["train"])
    lightning_cfg = dict(experiment_cfg.get("lightning", {}))

    pl.seed_everything(int(train_cfg.get("seed", 0)), workers=True)

    raw_model_cfg = dict(experiment_cfg["model"])
    lightning_module = RectangularConditionalJiTLightningModule(
        model_cfg=raw_model_cfg,
        objective_cfg=dict(experiment_cfg["objective"]),
        loss_cfg=dict(experiment_cfg["loss"]),
        optim_cfg=train_cfg,
        freeze_cfg=dict(experiment_cfg.get("freeze", {})),
        pretrained_cfg=dict(experiment_cfg.get("pretrained", {})),
    )
    datamodule = RectangularConditionalJiTDataModule(
        train_data_cfg=dict(experiment_cfg["data"]["train"]),
        val_data_cfg=dict(experiment_cfg["data"]["val"]) if experiment_cfg["data"].get("val") is not None else None,
        train_cfg=train_cfg,
        max_condition_channels=max(int(value) for value in raw_model_cfg["condition_channels_per_type"]),
    )

    output_dir = Path(experiment_cfg["output_dir"]) / experiment_cfg["experiment_name"]
    checkpoint_dir = output_dir / "checkpoints"
    logger = CSVLogger(save_dir=str(output_dir), name="csv_logs")
    callbacks = _build_callbacks(checkpoint_dir, enable_val_monitor=experiment_cfg["data"].get("val") is not None)
    precision = _resolve_precision(args.precision, lightning_cfg.get("precision", "32-true"))

    trainer_kwargs = {
        "default_root_dir": str(output_dir),
        "accelerator": _resolve_accelerator(args.device, lightning_cfg.get("accelerator", "gpu")),
        "devices": lightning_cfg.get("devices", 1),
        "precision": precision,
        "strategy": args.strategy or lightning_cfg.get("strategy", "auto"),
        "log_every_n_steps": int(lightning_cfg.get("log_every_n_steps", 10)),
        "max_steps": int(train_cfg["max_steps"]),
        "val_check_interval": lightning_cfg.get("val_check_interval", 200),
        "limit_train_batches": _resolve_limit(args.limit_train_batches, lightning_cfg.get("limit_train_batches")),
        "limit_val_batches": _resolve_limit(args.limit_val_batches, lightning_cfg.get("limit_val_batches")),
        "callbacks": callbacks,
        "logger": logger,
    }
    trainer_kwargs = {key: value for key, value in trainer_kwargs.items() if value is not None}

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=args.resume)


def _build_callbacks(checkpoint_dir: Path, enable_val_monitor: bool) -> list:
    callbacks = [
        ModelCheckpoint(dirpath=str(checkpoint_dir), save_last=True),
        LearningRateMonitor(logging_interval="step"),
    ]
    if enable_val_monitor:
        callbacks.insert(
            0,
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                save_top_k=1,
                monitor="val/loss_total",
                mode="min",
            ),
        )
    return callbacks


def _resolve_accelerator(device_arg: str | None, configured_accelerator: str) -> str:
    if device_arg is None:
        return str(configured_accelerator)
    return "gpu" if device_arg == "cuda" else str(device_arg)


def _resolve_limit(cli_value: str | None, config_value):
    value = cli_value if cli_value is not None else config_value
    if value is None:
        return None
    parsed = float(value)
    if parsed.is_integer():
        return int(parsed)
    return parsed


def _resolve_precision(cli_precision: str | None, configured_precision: str | None) -> str:
    precision = cli_precision if cli_precision is not None else configured_precision
    precision_str = str(precision).strip().lower()
    forbidden = {"16", "16-mixed", "bf16", "bf16-mixed", "mixed"}
    if precision_str in forbidden:
        raise ValueError("Mixed precision is disabled for this project. Use fp32 / 32-true only.")
    if precision_str not in {"32", "32-true", "fp32"}:
        raise ValueError("Only fp32 / 32-true precision is supported for this project.")
    return "32-true"


if __name__ == "__main__":
    main()