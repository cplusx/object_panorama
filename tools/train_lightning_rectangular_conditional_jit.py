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
    from lightning.pytorch.loggers import CSVLogger, WandbLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

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
    parser.add_argument(
        "-r",
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from checkpoint. Use -r or --resume with no value to resume from <output_dir>/<experiment_name>/checkpoints/last.ckpt; pass a path to resume from that checkpoint.",
    )
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
        validation_cfg=dict(experiment_cfg.get("validation", {})),
    )
    datamodule = RectangularConditionalJiTDataModule(
        train_data_cfg=dict(experiment_cfg["data"]["train"]),
        val_data_cfg=dict(experiment_cfg["data"]["val"]) if experiment_cfg["data"].get("val") is not None else None,
        train_cfg=train_cfg,
        validation_cfg=dict(experiment_cfg.get("validation", {})),
        max_condition_channels=max(int(value) for value in raw_model_cfg["condition_channels_per_type"]),
    )

    output_dir = Path(experiment_cfg["output_dir"]) / experiment_cfg["experiment_name"]
    checkpoint_dir = output_dir / "checkpoints"
    logger = _build_logger(experiment_cfg, output_dir)
    callbacks = _build_callbacks(checkpoint_dir, enable_val_monitor=experiment_cfg["data"].get("val") is not None)
    precision = _resolve_precision(args.precision, lightning_cfg.get("precision", "32-true"))

    trainer_kwargs = _build_trainer_kwargs(
        args=args,
        train_cfg=train_cfg,
        lightning_cfg=lightning_cfg,
        precision=precision,
        default_root_dir=str(output_dir),
        callbacks=callbacks,
        logger=logger,
        enable_validation=experiment_cfg["data"].get("val") is not None,
    )

    trainer = pl.Trainer(**trainer_kwargs)
    ckpt_path = _resolve_resume_checkpoint(args.resume, output_dir)
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)


def _build_callbacks(checkpoint_dir: Path, enable_val_monitor: bool) -> list:
    del enable_val_monitor
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="epoch_{epoch:06d}",
            every_n_epochs=1,
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    return callbacks


def _build_logger(experiment_cfg: dict, output_dir: Path):
    wandb_cfg = dict(experiment_cfg.get("wandb", {}))
    if bool(wandb_cfg.get("enabled", True)):
        return WandbLogger(
            project=wandb_cfg.get("project", "edge3d_flow"),
            name=wandb_cfg.get("name", experiment_cfg["experiment_name"]),
            save_dir=str(output_dir),
            tags=wandb_cfg.get("tags", []),
            log_model=bool(wandb_cfg.get("log_model", False)),
        )
    return CSVLogger(save_dir=str(output_dir), name="csv_logs")


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


def _resolve_resume_checkpoint(resume_arg, output_dir: Path):
    if resume_arg is None:
        return None
    if str(resume_arg).lower() == "auto":
        last_ckpt = (output_dir / "checkpoints" / "last.ckpt").expanduser().resolve()
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Requested auto resume, but last checkpoint does not exist: {last_ckpt}")
        return str(last_ckpt)

    resume_path = Path(resume_arg).expanduser().resolve()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    return str(resume_path)


def _build_trainer_kwargs(
    *,
    args: argparse.Namespace,
    train_cfg: dict,
    lightning_cfg: dict,
    precision: str,
    default_root_dir: str,
    callbacks: list,
    logger,
    enable_validation: bool,
) -> dict:
    accelerator = _resolve_accelerator(args.device, lightning_cfg.get("accelerator", "gpu"))
    devices = lightning_cfg.get("devices", 1)
    num_devices = _count_devices(devices)
    accumulate_grad_batches = _resolve_accumulate_grad_batches(
        batch_size=int(train_cfg["batch_size"]),
        effective_batch_size=int(train_cfg["effective_batch_size"]),
        num_devices=num_devices,
    )
    default_limit_train_batches = _resolve_micro_batches_per_epoch(
        effective_steps_per_epoch=int(train_cfg["effective_steps_per_epoch"]),
        accumulate_grad_batches=accumulate_grad_batches,
    )

    trainer_kwargs = {
        "default_root_dir": default_root_dir,
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
        "strategy": args.strategy or lightning_cfg.get("strategy", "deepspeed_stage_2"),
        "log_every_n_steps": int(lightning_cfg.get("log_every_n_steps", train_cfg.get("train_log_every_n_steps", 10))),
        "max_epochs": int(train_cfg["max_epochs"]),
        "accumulate_grad_batches": accumulate_grad_batches,
        "limit_train_batches": _resolve_limit(args.limit_train_batches, lightning_cfg.get("limit_train_batches"))
        or default_limit_train_batches,
        "limit_val_batches": _resolve_limit(args.limit_val_batches, lightning_cfg.get("limit_val_batches")),
        "check_val_every_n_epoch": int(lightning_cfg.get("check_val_every_n_epoch", 1)) if enable_validation else None,
        "callbacks": callbacks,
        "logger": logger,
        "use_distributed_sampler": False,
    }
    return {key: value for key, value in trainer_kwargs.items() if value is not None}


def _count_devices(devices) -> int:
    if isinstance(devices, int):
        return int(devices)
    if isinstance(devices, str):
        stripped = devices.strip()
        if stripped.isdigit():
            return int(stripped)
        raise ValueError(f"Unsupported devices specification: {devices!r}")
    if isinstance(devices, (list, tuple, set)):
        return len(devices)
    raise ValueError(f"Unsupported devices specification: {devices!r}")


def _resolve_accumulate_grad_batches(batch_size: int, effective_batch_size: int, num_devices: int) -> int:
    micro_batch = int(batch_size) * max(1, int(num_devices))
    total_batch = int(effective_batch_size)
    if total_batch % micro_batch != 0:
        raise ValueError(
            f"effective_batch_size={total_batch} must be divisible by batch_size*num_devices={micro_batch}"
        )
    return total_batch // micro_batch


def _resolve_micro_batches_per_epoch(effective_steps_per_epoch: int, accumulate_grad_batches: int) -> int:
    return int(effective_steps_per_epoch) * int(accumulate_grad_batches)


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