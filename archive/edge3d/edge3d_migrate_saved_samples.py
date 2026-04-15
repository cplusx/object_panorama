import argparse
from pathlib import Path

import numpy as np

from edge3d_tensor_format import is_mixed_precision_payload, save_mixed_precision_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy Edge3D tensor exports to the mixed-precision storage format.")
    parser.add_argument("--input-dir", default="/home/devdata/edge3d_data/equirectangular_data")
    parser.add_argument("--edge-max-hits", type=int, default=3)
    parser.add_argument("--pattern", default="*.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    migrated = 0
    skipped = 0

    for sample_path in sorted(input_dir.glob(args.pattern)):
        with np.load(sample_path, allow_pickle=False) as payload:
            if is_mixed_precision_payload(payload):
                skipped += 1
                continue
            uid = str(np.asarray(payload["uid"]).item())
            model_tensor = payload["model_tensor"]
            edge_tensor = payload["edge_tensor"]
            resolution = int(model_tensor.shape[1])
            model_max_hits = int(model_tensor.shape[0] // 7)

        save_mixed_precision_sample(
            sample_path,
            uid=uid,
            model_tensor=model_tensor,
            edge_tensor=edge_tensor,
            resolution=resolution,
            model_max_hits=model_max_hits,
            edge_max_hits=args.edge_max_hits,
        )
        migrated += 1
        print(f"migrated {sample_path.name}")

    print(f"migrated={migrated} skipped={skipped}")


if __name__ == "__main__":
    main()