from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge3d.tensor_format import load_sample_modalities
from pipeline import Edge3DX0BridgePipeline
from reconstruction import save_model_target_pred_pointclouds
from training.lightning_module import RectangularConditionalJiTLightningModule
from training.objectives import compute_prediction_losses
from utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample Edge3D x0-bridge inference from a training config and checkpoint.")
    parser.add_argument("config", help="Experiment YAML config")
    parser.add_argument(
        "--checkpoint",
        default="auto",
        help="Path to a Lightning/simple checkpoint file or DeepSpeed stage-2 checkpoint directory. Use 'auto' to load <output_dir>/<experiment_name>/checkpoints/last.ckpt.",
    )
    parser.add_argument("--sample-path", required=True, help="Path to a single Edge3D equirectangular NPZ sample")
    parser.add_argument("--output-dir", required=True, help="Directory where inference PLY outputs will be written")
    parser.add_argument("--num-steps", type=int, default=None, help="Override config inference.num_inference_steps")
    parser.add_argument("--cfg-scale", type=float, default=None, help="Override config inference.cfg_scale")
    parser.add_argument("--inference-dtype", default=None, help="Override config inference.inference_dtype")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def _flatten_hit_channels(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    if tensor.ndim == 3:
        return tensor.to(dtype=torch.float32)
    if tensor.ndim != 4:
        raise ValueError(f"Expected [hits, channels, H, W] or [hits, H, W], got {tuple(tensor.shape)}")
    hits, channels, height, width = tensor.shape
    return tensor.reshape(hits * channels, height, width).to(dtype=torch.float32)


def load_single_sample_batch(sample_path: str | Path) -> dict[str, torch.Tensor | list]:
    payload = load_sample_modalities(sample_path)
    return {
        "sample_ids": [str(payload["uid"])],
        "model_rgb": _flatten_hit_channels(payload["model_rgb"]).unsqueeze(0),
        "model_depth": torch.from_numpy(np.asarray(payload["model_depth"], dtype=np.float32)).unsqueeze(0),
        "model_normal": _flatten_hit_channels(payload["model_normal"]).unsqueeze(0),
        "edge_depth": torch.from_numpy(np.asarray(payload["edge_depth"], dtype=np.float32)).unsqueeze(0),
        "meta": [{"sample_path": str(Path(sample_path).expanduser().resolve())}],
    }


def _resolve_experiment_output_dir(config: dict[str, Any]) -> Path:
    configured_output_dir = Path(config["output_dir"]).expanduser()
    if not configured_output_dir.is_absolute():
        configured_output_dir = (REPO_ROOT / configured_output_dir).resolve()
    else:
        configured_output_dir = configured_output_dir.resolve()
    return configured_output_dir / str(config["experiment_name"])


def resolve_checkpoint_path(checkpoint_arg: str | Path | None, config: dict[str, Any]) -> Path | None:
    if checkpoint_arg is None:
        return None

    if str(checkpoint_arg).lower() == "auto":
        checkpoint_path = _resolve_experiment_output_dir(config) / "checkpoints" / "last.ckpt"
    else:
        checkpoint_path = Path(checkpoint_arg).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = checkpoint_path.resolve()
        else:
            checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Inference checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def resolve_inference_settings(
    config: dict[str, Any],
    *,
    num_steps: int | None = None,
    cfg_scale: float | None = None,
    inference_dtype: str | None = None,
) -> dict[str, Any]:
    resolved_cfg = dict(config.get("validation", {}))
    resolved_cfg.update(dict(config.get("inference", {})))

    resolved_num_steps = int(num_steps) if num_steps is not None else int(resolved_cfg.get("num_inference_steps", 20))
    resolved_cfg_scale = float(cfg_scale) if cfg_scale is not None else float(resolved_cfg.get("cfg_scale", 1.0))
    resolved_inference_dtype = str(inference_dtype) if inference_dtype is not None else str(resolved_cfg.get("inference_dtype", "float16"))

    if resolved_num_steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {resolved_num_steps}")
    if resolved_cfg_scale <= 0.0:
        raise ValueError(f"cfg_scale must be positive, got {resolved_cfg_scale}")

    return {
        "num_steps": resolved_num_steps,
        "cfg_scale": resolved_cfg_scale,
        "inference_dtype": resolved_inference_dtype,
    }


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    if checkpoint_path.is_dir():
        if not (checkpoint_path / "checkpoint").exists():
            raise ValueError(f"Unsupported checkpoint directory layout: {checkpoint_path}")
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        return get_fp32_state_dict_from_zero_checkpoint(str(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")
    return checkpoint


def _load_module_checkpoint(module: RectangularConditionalJiTLightningModule, checkpoint: dict[str, Any]) -> None:
    if "state_dict" in checkpoint:
        module.load_state_dict(checkpoint["state_dict"], strict=True)
        return
    if "model_state" in checkpoint:
        module.model.load_state_dict(checkpoint["model_state"], strict=True)
        return
    module.load_state_dict(checkpoint, strict=True)


def build_module(config: dict[str, Any], checkpoint_path: str | Path | None) -> RectangularConditionalJiTLightningModule:
    pretrained_cfg = dict(config.get("pretrained", {}))
    if checkpoint_path is not None:
        pretrained_cfg["load_jit"] = False

    module = RectangularConditionalJiTLightningModule(
        model_cfg=dict(config["model"]),
        objective_cfg=dict(config["objective"]),
        loss_cfg=dict(config["loss"]),
        optim_cfg=dict(config["train"]),
        freeze_cfg=dict(config.get("freeze", {})),
        pretrained_cfg=pretrained_cfg,
        validation_cfg=dict(config.get("validation", {})),
    )

    if checkpoint_path is None:
        return module

    checkpoint = _load_checkpoint_payload(Path(checkpoint_path))
    _load_module_checkpoint(module, checkpoint)
    return module


def _to_depth_layers_numpy(value: torch.Tensor | np.ndarray, expected_name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().to(dtype=torch.float32).numpy()
    else:
        array = np.asarray(value, dtype=np.float32)

    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Expected {expected_name} batch size 1, got shape {tuple(array.shape)}")
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"Expected {expected_name} shape [layers, H, W], got {tuple(array.shape)}")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def save_inference_pointclouds(
    output_dir: str | Path,
    *,
    model_depth: torch.Tensor | np.ndarray,
    pred_edge_depth: torch.Tensor | np.ndarray,
    target_edge_depth: torch.Tensor | np.ndarray,
) -> dict[str, str]:
    return save_model_target_pred_pointclouds(
        output_dir,
        model_depth=_to_depth_layers_numpy(model_depth, expected_name="model_depth"),
        pred_edge_depth=_to_depth_layers_numpy(pred_edge_depth, expected_name="pred_edge_depth"),
        target_edge_depth=_to_depth_layers_numpy(target_edge_depth, expected_name="target_edge_depth"),
    )


def run_single_sample_inference(
    config_path: str | Path,
    *,
    sample_path: str | Path,
    output_dir: str | Path,
    checkpoint: str | Path | None = "auto",
    device: str = "cuda",
    num_steps: int | None = None,
    cfg_scale: float | None = None,
    inference_dtype: str | None = None,
) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    resolved_checkpoint = resolve_checkpoint_path(checkpoint, config)
    inference_settings = resolve_inference_settings(
        config,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        inference_dtype=inference_dtype,
    )

    module = build_module(config, resolved_checkpoint)
    resolved_device = torch.device(device)
    model = module.model.to(device=resolved_device, dtype=torch.float32)
    model.eval()

    batch = load_single_sample_batch(sample_path)
    pipeline = Edge3DX0BridgePipeline(
        model,
        dict(config["objective"]),
        inference_dtype=inference_settings["inference_dtype"],
    )
    output = pipeline.generate(
        batch,
        num_steps=inference_settings["num_steps"],
        return_intermediates=False,
        cfg_scale=inference_settings["cfg_scale"],
    )

    pred_edge_depth = output["pred_edge_depth"].detach().cpu().to(dtype=torch.float32)
    target_edge_depth = torch.nan_to_num(batch["edge_depth"], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().to(dtype=torch.float32)
    model_depth = torch.nan_to_num(batch["model_depth"], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().to(dtype=torch.float32)
    pointcloud_paths = save_inference_pointclouds(
        output_dir,
        model_depth=model_depth,
        pred_edge_depth=pred_edge_depth,
        target_edge_depth=target_edge_depth,
    )

    loss_dict = compute_prediction_losses(pred_edge_depth, target_edge_depth, dict(config["loss"]))
    scalar_losses = {
        name: float(value.detach().cpu().item())
        for name, value in loss_dict.items()
        if isinstance(value, torch.Tensor) and value.numel() == 1
    }

    return {
        "checkpoint": None if resolved_checkpoint is None else str(resolved_checkpoint),
        "sample_path": str(Path(sample_path).expanduser().resolve()),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "num_steps": int(output["num_steps"]),
        "cfg_scale": float(output["cfg_scale"]),
        "condition_channels": int(output["condition_channels"]),
        "uses_cfg": bool(output["uses_cfg"]),
        "effective_inference_dtype": str(output["effective_inference_dtype"]),
        "losses": scalar_losses,
        "pointclouds": pointcloud_paths,
    }


def main() -> None:
    args = parse_args()
    result = run_single_sample_inference(
        args.config,
        sample_path=args.sample_path,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        inference_dtype=args.inference_dtype,
    )

    print(f"inference output: {result['output_dir']}")
    print(f"checkpoint: {result['checkpoint']}")
    print(f"condition channels: {result['condition_channels']}")
    print(f"num steps: {result['num_steps']}")
    print(f"cfg scale: {result['cfg_scale']}")
    print(f"uses cfg: {result['uses_cfg']}")
    print(f"effective inference dtype: {result['effective_inference_dtype']}")
    for name, value in sorted(result["losses"].items()):
        print(f"{name}: {value:.6f}")
    for name, path in sorted(result["pointclouds"].items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()