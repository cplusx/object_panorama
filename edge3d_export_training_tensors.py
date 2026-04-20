import argparse
import json
from pathlib import Path
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch

from edge3d_equirectangular_export import _build_training_representation
from edge3d_pipeline import (
    DEFAULT_EDGE3D_ALIGNMENT,
    MeshCanonicalizer,
    ObjaverseEdgeDataset,
    ObjaverseModelProvider,
    load_alignment_from_report,
)
from edge3d_tensor_format import (
    DEFAULT_EDGE_MAX_HITS,
    DEFAULT_MODEL_MAX_HITS,
    SAVE_FORMAT_VERSION,
    load_sample_modalities,
    save_mixed_precision_sample,
)
from inverse_spherical_representation import (
    mesh_to_inverse_spherical_representation,
    polylines_to_inverse_spherical_representation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-UID Edge3D training tensors as equirectangular arrays.")
    parser.add_argument("--dataset-root", default="/home/devdata/edge3d_data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--uids-json", default=None, help="Optional JSON file with a {'uids': [...]} payload")
    parser.add_argument("--alignment-report", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model-max-hits", type=int, default=DEFAULT_MODEL_MAX_HITS)
    parser.add_argument("--edge-max-hits", type=int, default=DEFAULT_EDGE_MAX_HITS)
    parser.add_argument("--max-hits", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--outer-radius", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--edge-color", default="0,190,255")
    parser.add_argument("--edge-sample-factor", type=float, default=2.0)
    parser.add_argument("--edge-depth-merge-tol", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-backend", default="auto", choices=["auto", "cpu_exact", "gpu_exact"])
    parser.add_argument("--shading", default="headlight", choices=["headlight", "none"])
    parser.add_argument("--download-processes", type=int, default=4)
    parser.add_argument("--download-timeout-sec", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    return parser.parse_args()


def _parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid RGB value: {value}")
    color = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(f"RGB channel out of range: {value}")
    return color


def _resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_backend(requested_backend: str, device: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    return "gpu_exact" if str(device).startswith("cuda") else "cpu_exact"


def _default_cache_dir() -> Path:
    return Path("/tmp/edge3d_objaverse_cache")


def _load_or_download_mesh(
    model_provider: ObjaverseModelProvider,
    uid: str,
    download_timeout_sec: float | None = None,
):
    try:
        return model_provider.load_mesh(uid)
    except FileNotFoundError:
        print(f"  mesh missing for {uid}, downloading on demand")
        model_provider.ensure_downloaded([uid], timeout_sec=download_timeout_sec)
        return model_provider.load_mesh(uid)


def _load_uids(
    dataset: ObjaverseEdgeDataset,
    uids_json: str | None,
    limit: int | None,
    start_idx: int,
    end_idx: int | None,
) -> list[str]:
    if uids_json is None:
        uids = list(dataset.available_ids)
    else:
        payload = json.loads(Path(uids_json).read_text(encoding="utf-8"))
        uids = [str(uid) for uid in payload["uids"]]
    start_idx = max(int(start_idx), 0)
    if end_idx is None:
        sliced = uids[start_idx:]
    else:
        sliced = uids[start_idx:max(int(end_idx), start_idx)]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def _edge_stats_from_tensor(edge_tensor: np.ndarray) -> dict[str, int | float]:
    valid = edge_tensor > 0
    layer_counts = valid.sum(axis=0)
    hit1_entries = int(valid[0].sum())
    total_entries = int(valid.sum())
    occupied_pixels = int((layer_counts > 0).sum())
    single_hit_pixels = int((layer_counts == 1).sum())
    multi_hit_pixels = int((layer_counts > 1).sum())
    return {
        "hit1_entries": hit1_entries,
        "total_entries_edge": total_entries,
        "hit1_over_total_entries_ratio": float(hit1_entries / max(total_entries, 1)),
        "occupied_pixels": occupied_pixels,
        "single_hit_pixels": single_hit_pixels,
        "multi_hit_pixels": multi_hit_pixels,
        "single_hit_pixel_ratio": float(single_hit_pixels / max(occupied_pixels, 1)),
        "multi_hit_pixel_ratio": float(multi_hit_pixels / max(occupied_pixels, 1)),
    }


def _load_existing_tensor_stats(tensor_path: Path) -> dict[str, object] | None:
    try:
        payload = load_sample_modalities(tensor_path, decode_model_normal=False)
    except Exception:
        return None
    edge_tensor = payload["edge_depth"]
    stats = _edge_stats_from_tensor(edge_tensor)
    return {
        "tensor_path": str(tensor_path),
        "storage_format_version": int(payload["storage_format_version"]),
        "model_component_shapes": payload["model_component_shapes"],
        "edge_shape": payload["edge_shape"],
        "storage_dtypes": payload["storage_dtypes"],
        **stats,
    }


def _write_summary(
    summary_path: Path,
    *,
    dataset_root: Path,
    output_dir: Path,
    start_idx: int,
    end_idx: int | None,
    resolution: int,
    model_max_hits: int,
    edge_max_hits: int,
    device: str,
    model_backend: str,
    reused_existing: int,
    failed_items: list[dict[str, object]],
    aggregate_hit1_entries: int,
    aggregate_total_entries: int,
    aggregate_occupied_pixels: int,
    aggregate_single_hit_pixels: int,
    aggregate_multi_hit_pixels: int,
    per_uid_stats: list[dict[str, object]],
) -> None:
    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "count": len(per_uid_stats),
        "reused_existing": reused_existing,
        "failed_count": len(failed_items),
        "failed_items": failed_items,
        "storage_format_version": SAVE_FORMAT_VERSION,
        "resolution": resolution,
        "width": resolution * 2,
        "model_max_hits": model_max_hits,
        "edge_max_hits": edge_max_hits,
        "device": device,
        "model_backend": model_backend,
        "tensor_layout": {
            "layout": "per_modality_per_hit",
            "model_components": {
                "model_rgb": {
                    "shape": [model_max_hits, 3, resolution, resolution * 2],
                    "storage_dtype": "float16",
                },
                "model_depth": {
                    "shape": [model_max_hits, resolution, resolution * 2],
                    "storage_dtype": "float16",
                },
                "model_normal": {
                    "shape": [model_max_hits, 3, resolution, resolution * 2],
                    "storage_dtype": "float8_e4m3fn_raw_uint8",
                },
            },
            "edge_components": {
                "edge_depth": {
                    "shape": [edge_max_hits, resolution, resolution * 2],
                    "storage_dtype": "float16",
                },
            },
        },
        "edge_hit_statistics": {
            "aggregate_hit1_entries": aggregate_hit1_entries,
            "aggregate_total_entries_edge": aggregate_total_entries,
            "aggregate_hit1_over_total_entries_ratio": float(aggregate_hit1_entries / max(aggregate_total_entries, 1)),
            "aggregate_occupied_pixels": aggregate_occupied_pixels,
            "aggregate_single_hit_pixels": aggregate_single_hit_pixels,
            "aggregate_multi_hit_pixels": aggregate_multi_hit_pixels,
            "aggregate_single_hit_pixel_ratio": float(aggregate_single_hit_pixels / max(aggregate_occupied_pixels, 1)),
            "aggregate_multi_hit_pixel_ratio": float(aggregate_multi_hit_pixels / max(aggregate_occupied_pixels, 1)),
        },
        "per_uid": per_uid_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_hits is not None:
        args.model_max_hits = int(args.max_hits)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "equirectangular_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment = load_alignment_from_report(args.alignment_report) if args.alignment_report else DEFAULT_EDGE3D_ALIGNMENT
    edge_color = _parse_rgb(args.edge_color)
    resolved_device = _resolve_device(args.device)
    resolved_model_backend = _resolve_model_backend(args.model_backend, resolved_device)
    if resolved_model_backend == "gpu_exact" and not str(resolved_device).startswith("cuda"):
        raise ValueError("gpu_exact backend requires a CUDA device")

    dataset = ObjaverseEdgeDataset(dataset_root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else _default_cache_dir()
    model_provider = ObjaverseModelProvider(cache_dir, download_processes=args.download_processes)
    canonicalizer = MeshCanonicalizer(alignment)
    uids = _load_uids(dataset, args.uids_json, args.limit, args.start_idx, args.end_idx)

    aggregate_hit1_entries = 0
    aggregate_total_entries = 0
    aggregate_occupied_pixels = 0
    aggregate_single_hit_pixels = 0
    aggregate_multi_hit_pixels = 0
    reused_existing = 0
    failed_items: list[dict[str, object]] = []
    per_uid_stats: list[dict[str, object]] = []
    summary_path = output_dir / "export_summary.json"

    for index, uid in enumerate(uids, start=1):
        start_time = time.perf_counter()
        tensor_path = output_dir / f"{uid}.npz"
        if tensor_path.exists() and not args.overwrite:
            existing = _load_existing_tensor_stats(tensor_path)
            if existing is not None:
                aggregate_hit1_entries += int(existing["hit1_entries"])
                aggregate_total_entries += int(existing["total_entries_edge"])
                aggregate_occupied_pixels += int(existing["occupied_pixels"])
                aggregate_single_hit_pixels += int(existing["single_hit_pixels"])
                aggregate_multi_hit_pixels += int(existing["multi_hit_pixels"])
                per_uid_stats.append(
                    {
                        "uid": uid,
                        **existing,
                        "elapsed_sec": 0.0,
                        "source": "existing",
                    }
                )
                reused_existing += 1
                print(f"[{index:05d}/{len(uids):05d}] {uid} exists, counted without recompute")
                _write_summary(
                    summary_path,
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    start_idx=args.start_idx,
                    end_idx=args.end_idx,
                    resolution=args.resolution,
                    model_max_hits=args.model_max_hits,
                    edge_max_hits=args.edge_max_hits,
                    device=resolved_device,
                    model_backend=resolved_model_backend,
                    reused_existing=reused_existing,
                    failed_items=failed_items,
                    aggregate_hit1_entries=aggregate_hit1_entries,
                    aggregate_total_entries=aggregate_total_entries,
                    aggregate_occupied_pixels=aggregate_occupied_pixels,
                    aggregate_single_hit_pixels=aggregate_single_hit_pixels,
                    aggregate_multi_hit_pixels=aggregate_multi_hit_pixels,
                    per_uid_stats=per_uid_stats,
                )
                continue
            print(f"[{index:05d}/{len(uids):05d}] {uid} exists but unreadable, recomputing")

        try:
            mesh = _load_or_download_mesh(model_provider, uid, download_timeout_sec=args.download_timeout_sec)
            canonical_mesh = canonicalizer.canonicalize_mesh(mesh)
            edge_polylines = dataset.load_edge_polylines(uid).astype(np.float32)

            model_rep = mesh_to_inverse_spherical_representation(
                canonical_mesh,
                resolution=args.resolution,
                max_hits=args.model_max_hits,
                outer_radius=args.outer_radius,
                batch_size=args.batch_size,
                shading=args.shading,
                device=resolved_device,
                stop_at_origin=True,
                backend=resolved_model_backend,
            )
            edge_rep = polylines_to_inverse_spherical_representation(
                edge_polylines,
                resolution=args.resolution,
                max_hits=args.edge_max_hits,
                edge_color=edge_color,
                sample_factor=args.edge_sample_factor,
                depth_merge_tol=args.edge_depth_merge_tol,
                device=resolved_device,
            )

            model_tensor, _, _ = _build_training_representation(model_rep, include_rgb=True, include_normal=True)
            edge_tensor, _, _ = _build_training_representation(edge_rep, include_rgb=False, include_normal=False)
            format_info = save_mixed_precision_sample(
                tensor_path,
                uid=uid,
                model_tensor=model_tensor.astype(np.float32),
                edge_tensor=edge_tensor.astype(np.float32),
                resolution=args.resolution,
                model_max_hits=args.model_max_hits,
                edge_max_hits=args.edge_max_hits,
            )

            stats = _edge_stats_from_tensor(edge_tensor)
            aggregate_hit1_entries += int(stats["hit1_entries"])
            aggregate_total_entries += int(stats["total_entries_edge"])
            aggregate_occupied_pixels += int(stats["occupied_pixels"])
            aggregate_single_hit_pixels += int(stats["single_hit_pixels"])
            aggregate_multi_hit_pixels += int(stats["multi_hit_pixels"])

            per_uid_stats.append(
                {
                    "uid": uid,
                    "tensor_path": str(tensor_path),
                    **format_info,
                    **stats,
                    "elapsed_sec": round(time.perf_counter() - start_time, 3),
                    "source": "computed",
                }
            )
            print(
                f"[{index:05d}/{len(uids):05d}] {uid} saved model_rgb={tuple(format_info['model_rgb_shape'])} edge_depth={tuple(format_info['edge_depth_shape'])} "
                f"hit1/edge_entries={stats['hit1_entries']}/{stats['total_entries_edge']} ({stats['hit1_over_total_entries_ratio']:.4f}) "
                f"single_pixel_ratio={stats['single_hit_pixel_ratio']:.4f}"
            )
        except Exception as exc:
            failed_items.append(
                {
                    "uid": uid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[{index:05d}/{len(uids):05d}] {uid} failed, skipping: {type(exc).__name__}: {exc}")
        finally:
            model_provider.delete_cached_mesh(uid)

        _write_summary(
            summary_path,
            dataset_root=dataset_root,
            output_dir=output_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            resolution=args.resolution,
            model_max_hits=args.model_max_hits,
            edge_max_hits=args.edge_max_hits,
            device=resolved_device,
            model_backend=resolved_model_backend,
            reused_existing=reused_existing,
            failed_items=failed_items,
            aggregate_hit1_entries=aggregate_hit1_entries,
            aggregate_total_entries=aggregate_total_entries,
            aggregate_occupied_pixels=aggregate_occupied_pixels,
            aggregate_single_hit_pixels=aggregate_single_hit_pixels,
            aggregate_multi_hit_pixels=aggregate_multi_hit_pixels,
            per_uid_stats=per_uid_stats,
        )

    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()