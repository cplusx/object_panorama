import argparse
import json
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models import JiTModel
from models.jit_checkpoint_loader import (
    infer_jit_model_config_from_checkpoint,
    instantiate_model_from_public_checkpoint,
    load_public_jit_checkpoint,
    load_public_weights_into_model,
)
from pipelines import JiTPipeline
from schedulers import JiTScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run class-conditional JiT image sampling through the local Diffusers pipeline.")
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--scheduler-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--attn-dropout", type=float, default=None)
    parser.add_argument("--proj-dropout", type=float, default=None)
    parser.add_argument("--variant", default="ema1", choices=["raw", "ema1", "ema2"])
    parser.add_argument("--class-labels", required=True, help="Comma-separated class labels, for example 0,1,2")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--interval-min", type=float, default=None)
    parser.add_argument("--interval-max", type=float, default=None)
    parser.add_argument("--sampling-method", default=None, choices=["euler", "heun", None])
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--t-eps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="jit_samples")
    return parser.parse_args()


def _parse_class_labels(value: str) -> list[int]:
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one class label is required")
    return [int(item) for item in labels]


def _load_model_and_scheduler(args: argparse.Namespace) -> tuple[JiTModel, JiTScheduler, dict]:
    metadata: dict[str, object] = {}
    if args.bundle_dir is not None:
        bundle_dir = Path(args.bundle_dir)
        model = JiTModel.from_pretrained(bundle_dir / "transformer")
        scheduler_path = bundle_dir / "scheduler"
        scheduler = JiTScheduler.from_pretrained(scheduler_path) if scheduler_path.exists() else JiTScheduler()
        metadata["source"] = "bundle"
        metadata["bundle_dir"] = str(bundle_dir.resolve())
        return model, scheduler, metadata

    if args.model_dir is not None:
        model = JiTModel.from_pretrained(args.model_dir)
        scheduler = JiTScheduler.from_pretrained(args.scheduler_dir) if args.scheduler_dir else JiTScheduler()
        metadata["source"] = "pretrained_dirs"
        metadata["model_dir"] = str(Path(args.model_dir).resolve())
        metadata["scheduler_dir"] = str(Path(args.scheduler_dir).resolve()) if args.scheduler_dir else None
        return model, scheduler, metadata

    if args.checkpoint is None:
        raise ValueError("Provide one of --bundle-dir, --model-dir, or --checkpoint")

    checkpoint = load_public_jit_checkpoint(args.checkpoint, map_location="cpu")
    inferred = infer_jit_model_config_from_checkpoint(checkpoint)
    model, checkpoint, inferred = instantiate_model_from_public_checkpoint(
        checkpoint,
        preset_name=args.preset,
        input_size=args.image_size,
        num_classes=args.num_classes,
        attn_drop=args.attn_dropout,
        proj_drop=args.proj_dropout,
    )
    load_report = load_public_weights_into_model(model, checkpoint, variant=args.variant, strict=False)
    scheduler = JiTScheduler(
        sampling_method=args.sampling_method or inferred.get("sampling_method", "heun"),
        t_eps=float(args.t_eps if args.t_eps is not None else inferred.get("t_eps", 5e-2)),
        noise_scale=float(args.noise_scale if args.noise_scale is not None else inferred.get("noise_scale", 1.0)),
        cfg_interval_min=float(args.interval_min if args.interval_min is not None else inferred.get("interval_min", 0.0)),
        cfg_interval_max=float(args.interval_max if args.interval_max is not None else inferred.get("interval_max", 1.0)),
    )
    metadata.update(
        {
            "source": "public_checkpoint",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "inferred": inferred,
            "load_report": load_report,
        }
    )
    return model, scheduler, metadata


def main() -> None:
    args = parse_args()
    class_labels = _parse_class_labels(args.class_labels)
    model, scheduler, metadata = _load_model_and_scheduler(args)
    device = torch.device(args.device)
    model = model.to(device)
    pipeline = JiTPipeline(transformer=model, scheduler=scheduler).to(device)

    outputs = pipeline(
        class_labels=class_labels,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        interval_min=args.interval_min,
        interval_max=args.interval_max,
        sampling_method=args.sampling_method,
        seed=args.seed,
        output_type="pil",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for index, (label, image) in enumerate(zip(class_labels, outputs.images)):
        image_path = output_dir / f"sample_{index:05d}_class{label:04d}.png"
        image.save(image_path)
        saved_paths.append(str(image_path.resolve()))

    report = {
        "output_dir": str(output_dir.resolve()),
        "saved_paths": saved_paths,
        "class_labels": class_labels,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "sampling_method": args.sampling_method or scheduler.config.sampling_method,
        "seed": args.seed,
        "device": str(device),
        "source": metadata,
    }
    (output_dir / "sample_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()