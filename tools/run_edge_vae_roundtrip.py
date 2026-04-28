from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_vae.configs import apply_edge_vae_overrides, load_edge_vae_config
from edge_vae.roundtrip import run_edge_vae_roundtrip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDXL VAE roundtrip on edge_depth in raw or DF-coded space.")
    parser.add_argument("--config", default=None, help="Optional YAML config path under configs/vae")
    parser.add_argument("--sample-path", required=True, help="Path to an Edge3D .npz sample")
    parser.add_argument("--output-dir", required=True, help="Directory to write roundtrip outputs")
    parser.add_argument("--mode", required=True, choices=["raw", "df"], help="Roundtrip mode")
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--raw-scale", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--valid-eps", type=float, default=None)
    parser.add_argument("--decode-valid-threshold", type=float, default=None)
    parser.add_argument("--vae", dest="vae_name", default=None, help="diffusers AutoencoderKL repo or local path")
    parser.add_argument("--torch-dtype", default=None, help="float16 or float32")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_edge_vae_config(args.config, mode=args.mode)
    config = apply_edge_vae_overrides(
        config,
        vae_name=args.vae_name,
        torch_dtype=args.torch_dtype,
        device=args.device,
        depth_scale=args.depth_scale,
        raw_scale=args.raw_scale,
        beta=args.beta,
        valid_eps=args.valid_eps,
        decode_valid_threshold=args.decode_valid_threshold,
    )

    result = run_edge_vae_roundtrip(
        sample_path=args.sample_path,
        output_dir=args.output_dir,
        mode=args.mode,
        vae_cfg=dict(config["vae"]),
        transform_cfg=dict(config["transform"]),
        runtime_cfg=dict(config["runtime"]),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()