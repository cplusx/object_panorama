import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from edge3d.tensor_format import load_sample_modalities


@dataclass
class RunningStats:
    count: int = 0
    finite_count: int = 0
    nonfinite_count: int = 0
    sum_abs_error: float = 0.0
    sum_sq_error: float = 0.0
    max_abs_error: float = 0.0
    sum_abs_value: float = 0.0
    max_abs_value: float = 0.0

    def finalize(self) -> dict[str, float | int | None]:
        if self.finite_count == 0:
            return {
                "count": self.count,
                "finite_count": self.finite_count,
                "nonfinite_count": self.nonfinite_count,
                "mae": None,
                "rmse": None,
                "max_abs": None,
                "mean_abs_value": None,
                "max_abs_value": None,
                "mae_over_mean_abs_value": None,
            }
        mae = self.sum_abs_error / self.finite_count
        mean_abs_value = self.sum_abs_value / self.finite_count
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "mae": mae,
            "rmse": (self.sum_sq_error / self.finite_count) ** 0.5,
            "max_abs": self.max_abs_error,
            "mean_abs_value": mean_abs_value,
            "max_abs_value": self.max_abs_value,
            "mae_over_mean_abs_value": mae / max(mean_abs_value, 1e-12),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Edge3D tensor quantization error across stored .npz exports.")
    parser.add_argument("--input-dir", default="/home/devdata/edge3d_data/equirectangular_data")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtypes", default="fp16,float8_e4m3fn,float8_e5m2")
    return parser.parse_args()


def _resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype_name_to_torch(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float8_e4m3fnuz": torch.float8_e4m3fnuz,
        "float8_e5m2fnuz": torch.float8_e5m2fnuz,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[name]


def _update_running_stats(values: np.ndarray, dtype: torch.dtype, device: str, stats: RunningStats, chunk_size: int = 4_000_000) -> None:
    if values.size == 0:
        return
    flat_values = np.ascontiguousarray(values.reshape(-1).astype(np.float32, copy=False))
    stats.count += int(flat_values.size)
    for start in range(0, flat_values.size, chunk_size):
        chunk_np = flat_values[start:start + chunk_size]
        chunk = torch.from_numpy(chunk_np).to(device=device, dtype=torch.float32)
        roundtrip = chunk.to(dtype).to(torch.float32)
        finite_mask = torch.isfinite(roundtrip)
        stats.nonfinite_count += int((~finite_mask).sum().item())
        if not bool(finite_mask.any()):
            continue
        safe_source = chunk[finite_mask]
        safe_roundtrip = roundtrip[finite_mask]
        abs_error = (safe_roundtrip - safe_source).abs()
        stats.finite_count += int(abs_error.numel())
        stats.sum_abs_error += float(abs_error.sum().item())
        stats.sum_sq_error += float(abs_error.square().sum().item())
        stats.max_abs_error = max(stats.max_abs_error, float(abs_error.max().item()))
        abs_source = safe_source.abs()
        stats.sum_abs_value += float(abs_source.sum().item())
        stats.max_abs_value = max(stats.max_abs_value, float(abs_source.max().item()))


def _extract_model_rgb(model_rgb: np.ndarray, model_depth: np.ndarray) -> np.ndarray:
    valid = model_depth > 0
    rgb_values = [model_rgb[:, channel][valid] for channel in range(model_rgb.shape[1])]
    return np.concatenate(rgb_values, axis=0) if rgb_values else np.empty((0,), dtype=np.float32)


def _extract_model_depth(model_depth: np.ndarray) -> np.ndarray:
    return model_depth[model_depth > 0]


def _extract_model_normal(model_normal: np.ndarray, model_depth: np.ndarray) -> np.ndarray:
    valid = model_depth > 0
    normal_values = [model_normal[:, channel][valid] for channel in range(model_normal.shape[1])]
    return np.concatenate(normal_values, axis=0) if normal_values else np.empty((0,), dtype=np.float32)


def _extract_edge_depth(edge_depth: np.ndarray) -> np.ndarray:
    return edge_depth[edge_depth > 0]


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_json = Path(args.output_json) if args.output_json else input_dir / "analysis_quantization.json"
    summary_json = Path(args.summary_json) if args.summary_json else input_dir / "export_summary.json"
    device = _resolve_device(args.device)
    dtype_names = [item.strip() for item in args.dtypes.split(",") if item.strip()]
    dtype_mapping = {name: _dtype_name_to_torch(name) for name in dtype_names}

    source_summary = None
    if summary_json.exists():
        source_summary = json.loads(summary_json.read_text(encoding="utf-8"))

    npz_paths = sorted(input_dir.glob("*.npz"))
    failed_reads: list[dict[str, str]] = []

    cumulative_layer_entries: list[int] = []
    total_entries_edge = 0
    modality_names = ["model_rgb", "model_depth", "model_normal", "edge_depth"]
    quant_stats = {
        dtype_name: {modality: RunningStats() for modality in modality_names}
        for dtype_name in dtype_mapping
    }

    for npz_path in npz_paths:
        try:
            payload = load_sample_modalities(npz_path, decode_model_normal=True)
        except Exception as exc:
            failed_reads.append({"path": str(npz_path), "error": str(exc)})
            continue

        model_rgb = payload["model_rgb"]
        model_depth = payload["model_depth"]
        model_normal = payload["model_normal"]
        edge_depth = payload["edge_depth"]

        edge_valid = edge_depth > 0
        if not cumulative_layer_entries:
            cumulative_layer_entries = [0] * edge_depth.shape[0]
        for index in range(edge_depth.shape[0]):
            cumulative_layer_entries[index] += int(edge_valid[: index + 1].sum())
        total_entries_edge += int(edge_valid.sum())

        extracted = {
            "model_rgb": _extract_model_rgb(model_rgb, model_depth),
            "model_depth": _extract_model_depth(model_depth),
            "model_normal": _extract_model_normal(model_normal, model_depth),
            "edge_depth": _extract_edge_depth(edge_depth),
        }
        for dtype_name, torch_dtype in dtype_mapping.items():
            for modality, values in extracted.items():
                _update_running_stats(values, torch_dtype, device, quant_stats[dtype_name][modality])

    edge_hit_retention = {
        f"cum_{index + 1}_hit_over_saved_edge": cumulative_layer_entries[index] / max(total_entries_edge, 1)
        for index in range(len(cumulative_layer_entries))
    }

    report = {
        "analysis_device": device,
        "source_summary": source_summary,
        "loaded_npz_count": len(npz_paths) - len(failed_reads),
        "failed_read_count": len(failed_reads),
        "failed_reads": failed_reads,
        "edge_hit_retention_vs_5hit": edge_hit_retention,
        "quantization_roundtrip": {
            dtype_name: {
                modality: stats.finalize()
                for modality, stats in modality_stats.items()
            }
            for dtype_name, modality_stats in quant_stats.items()
        },
    }
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()