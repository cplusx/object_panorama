from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge3d.tensor_format import load_sample_modalities
from pipeline import Edge3DX0BridgePipeline
from training.lightning_module import RectangularConditionalJiTLightningModule
from utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-sample Edge3D x0-bridge inference from a training config and checkpoint.")
    parser.add_argument("config", help="Experiment YAML config")
    parser.add_argument("--checkpoint", default=None, help="Path to a Lightning or simple-training checkpoint")
    parser.add_argument("--sample-path", required=True, help="Path to a single Edge3D equirectangular NPZ sample")
    parser.add_argument("--output-dir", required=True, help="Directory where inference outputs will be written")
    parser.add_argument("--num-steps", type=int, default=20)
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


def build_module(config: dict, checkpoint_path: str | Path | None):
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

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        module.load_state_dict(checkpoint["state_dict"], strict=True)
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        module.model.load_state_dict(checkpoint["model_state"], strict=True)
    elif isinstance(checkpoint, dict):
        module.load_state_dict(checkpoint, strict=True)
    else:
        raise ValueError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")
    return module


def _normalize_panel(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image, dtype=np.float32)
    valid = image[finite]
    lo = float(valid.min())
    hi = float(valid.max())
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    normalized = np.clip((np.nan_to_num(image, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return normalized.astype(np.float32)


def save_preview(output_dir: Path, target: torch.Tensor, pred: torch.Tensor) -> None:
    target_hit0 = target[0, 0].detach().cpu().numpy()
    pred_hit0 = pred[0, 0].detach().cpu().numpy()
    error_hit0 = np.abs(target_hit0 - pred_hit0)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = [
        (_normalize_panel(target_hit0), "target edge depth hit0"),
        (_normalize_panel(pred_hit0), "pred edge depth hit0"),
        (_normalize_panel(error_hit0), "abs error hit0"),
    ]
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image, cmap="magma")
        axis.set_title(title)
        axis.axis("off")
    figure.savefig(output_dir / "preview.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    module = build_module(config, args.checkpoint)
    if args.checkpoint is None and not bool(config.get("pretrained", {}).get("load_jit", False)):
        print("warning: running inference with random weights because no checkpoint was provided and pretrained.load_jit=false", file=sys.stderr)
    device = torch.device(args.device)
    model = module.model.to(device=device, dtype=torch.float32)
    model.eval()

    batch = load_single_sample_batch(args.sample_path)
    pipeline = Edge3DX0BridgePipeline(
        model,
        dict(config["objective"]),
        inference_dtype=str(config.get("validation", {}).get("inference_dtype", "float16")),
    )
    output = pipeline.generate(batch, num_steps=args.num_steps, return_intermediates=False)

    pred_edge_depth = output["pred_edge_depth"].detach().cpu().to(dtype=torch.float32)
    target_edge_depth = torch.nan_to_num(batch["edge_depth"], nan=0.0, posinf=0.0, neginf=0.0).detach().cpu().to(dtype=torch.float32)
    initial_noise = output["initial_noise"].detach().cpu().to(dtype=torch.float32)

    torch.save(pred_edge_depth, output_dir / "pred_edge_depth.pt")
    torch.save(target_edge_depth, output_dir / "target_edge_depth.pt")
    torch.save(initial_noise, output_dir / "initial_noise.pt")
    save_preview(output_dir, target_edge_depth, pred_edge_depth)

    print(f"inference output: {output_dir}")
    print(f"condition channels: {output['condition_channels']}")
    print(f"num steps: {output['num_steps']}")


if __name__ == "__main__":
    main()