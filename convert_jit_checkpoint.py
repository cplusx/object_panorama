import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from models.jit_checkpoint_loader import (
    infer_jit_model_config_from_checkpoint,
    instantiate_model_from_public_checkpoint,
    load_public_jit_checkpoint,
    load_public_weights_into_model,
    save_diffusers_jit_bundle,
)
from schedulers import JiTScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an official public JiT checkpoint into local Diffusers format.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--attn-dropout", type=float, default=None)
    parser.add_argument("--proj-dropout", type=float, default=None)
    parser.add_argument("--variant", default="ema1", choices=["raw", "ema1", "ema2"])
    parser.add_argument("--sampling-method", default=None, choices=["euler", "heun", None])
    parser.add_argument("--interval-min", type=float, default=None)
    parser.add_argument("--interval-max", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--t-eps", type=float, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    load_report = load_public_weights_into_model(model, checkpoint, variant=args.variant, strict=args.strict)

    scheduler = JiTScheduler(
        sampling_method=args.sampling_method or inferred.get("sampling_method", "heun"),
        t_eps=float(args.t_eps if args.t_eps is not None else inferred.get("t_eps", 5e-2)),
        noise_scale=float(args.noise_scale if args.noise_scale is not None else inferred.get("noise_scale", 1.0)),
        cfg_interval_min=float(args.interval_min if args.interval_min is not None else inferred.get("interval_min", 0.0)),
        cfg_interval_max=float(args.interval_max if args.interval_max is not None else inferred.get("interval_max", 1.0)),
    )

    report = {
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "inferred": inferred,
        "load_report": load_report,
    }
    bundle_paths = save_diffusers_jit_bundle(args.output_dir, model, scheduler=scheduler, metadata=report)
    report.update(bundle_paths)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()