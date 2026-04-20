import argparse
import multiprocessing as mp
from pathlib import Path
import queue
import time

import numpy as np
import torch

from edge3d.generation.equirectangular_export import _build_training_representation
from edge3d.generation.export_training_tensors import (
    _default_cache_dir,
    _edge_stats_from_tensor,
    _load_existing_tensor_stats,
    _load_or_download_mesh,
    _load_uids,
    _parse_rgb,
    _resolve_model_backend,
    _write_summary,
)
from edge3d.generation.pipeline import (
    DEFAULT_EDGE3D_ALIGNMENT,
    MeshCanonicalizer,
    ObjaverseEdgeDataset,
    ObjaverseModelProvider,
    load_alignment_from_report,
)
from edge3d.representation.layered_spherical_representation import (
    mesh_to_layered_spherical_representation,
    polylines_to_layered_spherical_representation,
)
from edge3d.tensor_format import DEFAULT_EDGE_MAX_HITS, DEFAULT_MODEL_MAX_HITS, save_mixed_precision_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel Edge3D equirectangular tensor export with multiple GPU workers.")
    parser.add_argument("--dataset-root", default="/home/devdata/edge3d_data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--uids-json", default=None)
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
    parser.add_argument("--model-backend", default="gpu_exact", choices=["auto", "cpu_exact", "gpu_exact"])
    parser.add_argument("--shading", default="headlight", choices=["headlight", "none"])
    parser.add_argument("--download-processes", type=int, default=1)
    parser.add_argument("--download-timeout-sec", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--gpu-ids", default="0,1", help="Comma-separated CUDA device indices, e.g. 0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--summary-every", type=int, default=10)
    return parser.parse_args()


def _worker_devices(gpu_ids: list[int], workers_per_gpu: int) -> list[str]:
    devices: list[str] = []
    for gpu_id in gpu_ids:
        for _ in range(workers_per_gpu):
            devices.append(f"cuda:{gpu_id}")
    return devices


def _prime_objaverse_cache(cache_dir: Path, uid: str, download_timeout_sec: float) -> None:
    if (cache_dir / "hf-objaverse-v1").exists():
        return
    print(f"Priming Objaverse cache metadata in {cache_dir} using {uid}")
    model_provider = ObjaverseModelProvider(cache_dir, download_processes=1)
    model_provider.ensure_downloaded([uid], timeout_sec=download_timeout_sec)
    model_provider.delete_cached_mesh(uid)


def _worker_loop(
    worker_name: str,
    device: str,
    dataset_root: str,
    cache_dir: str,
    output_dir: str,
    alignment_payload: dict,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    resolution: int,
    model_max_hits: int,
    edge_max_hits: int,
    outer_radius: float,
    batch_size: int,
    edge_color: tuple[int, int, int],
    edge_sample_factor: float,
    edge_depth_merge_tol: float | None,
    model_backend: str,
    shading: str,
    download_processes: int,
    download_timeout_sec: float,
) -> None:
    gpu_index = int(device.split(":", 1)[1])
    torch.cuda.set_device(gpu_index)

    dataset = ObjaverseEdgeDataset(dataset_root)
    model_provider = ObjaverseModelProvider(Path(cache_dir), download_processes=download_processes)
    alignment = DEFAULT_EDGE3D_ALIGNMENT.__class__(
        axis_order=tuple(alignment_payload["axis_order"]),
        axis_signs=tuple(alignment_payload["axis_signs"]),
        normalization=str(alignment_payload["normalization"]),
    )
    canonicalizer = MeshCanonicalizer(alignment)
    output_root = Path(output_dir)

    while True:
        task = task_queue.get()
        if task is None:
            break
        index, uid = task
        start_time = time.perf_counter()
        tensor_path = output_root / f"{uid}.npz"
        try:
            mesh = _load_or_download_mesh(model_provider, uid, download_timeout_sec=download_timeout_sec)
            canonical_mesh = canonicalizer.canonicalize_mesh(mesh)
            edge_polylines = dataset.load_edge_polylines(uid).astype(np.float32)

            model_rep = mesh_to_layered_spherical_representation(
                canonical_mesh,
                resolution=resolution,
                max_hits=model_max_hits,
                outer_radius=outer_radius,
                batch_size=batch_size,
                shading=shading,
                device=device,
                stop_at_origin=True,
                backend=model_backend,
            )
            edge_rep = polylines_to_layered_spherical_representation(
                edge_polylines,
                resolution=resolution,
                max_hits=edge_max_hits,
                edge_color=edge_color,
                sample_factor=edge_sample_factor,
                depth_merge_tol=edge_depth_merge_tol,
                device=device,
            )

            model_tensor, _, _ = _build_training_representation(model_rep, include_rgb=True, include_normal=True)
            edge_tensor, _, _ = _build_training_representation(edge_rep, include_rgb=False, include_normal=False)
            format_info = save_mixed_precision_sample(
                tensor_path,
                uid=uid,
                model_tensor=model_tensor.astype(np.float32),
                edge_tensor=edge_tensor.astype(np.float32),
                resolution=resolution,
                model_max_hits=model_max_hits,
                edge_max_hits=edge_max_hits,
            )
            stats = _edge_stats_from_tensor(edge_tensor)
            result_queue.put(
                {
                    "type": "success",
                    "index": index,
                    "uid": uid,
                    "worker": worker_name,
                    "device": device,
                    "tensor_path": str(tensor_path),
                    **format_info,
                    **stats,
                    "elapsed_sec": round(time.perf_counter() - start_time, 3),
                    "source": "computed",
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "type": "failure",
                    "index": index,
                    "uid": uid,
                    "worker": worker_name,
                    "device": device,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        finally:
            model_provider.delete_cached_mesh(uid)


def main() -> None:
    args = parse_args()
    if args.max_hits is not None:
        args.model_max_hits = int(args.max_hits)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "equirectangular_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else _default_cache_dir()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the parallel GPU exporter")

    gpu_ids = [int(part.strip()) for part in args.gpu_ids.split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")
    worker_devices = _worker_devices(gpu_ids, args.workers_per_gpu)

    alignment = load_alignment_from_report(args.alignment_report) if args.alignment_report else DEFAULT_EDGE3D_ALIGNMENT
    alignment_payload = {
        "axis_order": list(alignment.axis_order),
        "axis_signs": list(alignment.axis_signs),
        "normalization": alignment.normalization,
    }
    edge_color = _parse_rgb(args.edge_color)

    dataset = ObjaverseEdgeDataset(dataset_root)
    uids = _load_uids(dataset, args.uids_json, args.limit, args.start_idx, args.end_idx)

    aggregate_hit1_entries = 0
    aggregate_total_entries = 0
    aggregate_occupied_pixels = 0
    aggregate_single_hit_pixels = 0
    aggregate_multi_hit_pixels = 0
    reused_existing = 0
    failed_items: list[dict[str, object]] = []
    results_by_index: dict[int, dict[str, object]] = {}

    tasks: list[tuple[int, str]] = []
    for local_index, uid in enumerate(uids, start=1):
        tensor_path = output_dir / f"{uid}.npz"
        if tensor_path.exists() and not args.overwrite:
            continue
        tasks.append((local_index, uid))

    summary_path = output_dir / "export_summary.json"
    _write_summary(
        summary_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        resolution=args.resolution,
        model_max_hits=args.model_max_hits,
        edge_max_hits=args.edge_max_hits,
        device=",".join(worker_devices),
        model_backend=_resolve_model_backend(args.model_backend, "cuda:0"),
        reused_existing=reused_existing,
        failed_items=failed_items,
        aggregate_hit1_entries=aggregate_hit1_entries,
        aggregate_total_entries=aggregate_total_entries,
        aggregate_occupied_pixels=aggregate_occupied_pixels,
        aggregate_single_hit_pixels=aggregate_single_hit_pixels,
        aggregate_multi_hit_pixels=aggregate_multi_hit_pixels,
        per_uid_stats=[results_by_index[i] for i in sorted(results_by_index)],
    )

    if not tasks:
        print("No compute tasks remain after reusing existing tensors")
        print(f"wrote summary: {summary_path}")
        return

    _prime_objaverse_cache(cache_dir, tasks[0][1], args.download_timeout_sec)

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    workers: list[mp.Process] = []
    resolved_model_backend = _resolve_model_backend(args.model_backend, "cuda:0")

    for task in tasks:
        task_queue.put(task)
    for _ in worker_devices:
        task_queue.put(None)

    for worker_index, device in enumerate(worker_devices):
        worker_name = f"worker{worker_index}@{device}"
        proc = ctx.Process(
            target=_worker_loop,
            args=(
                worker_name,
                device,
                str(dataset_root),
                str(cache_dir),
                str(output_dir),
                alignment_payload,
                task_queue,
                result_queue,
                args.resolution,
                args.model_max_hits,
                args.edge_max_hits,
                args.outer_radius,
                args.batch_size,
                edge_color,
                args.edge_sample_factor,
                args.edge_depth_merge_tol,
                resolved_model_backend,
                args.shading,
                args.download_processes,
                args.download_timeout_sec,
            ),
        )
        proc.start()
        workers.append(proc)

    completed = 0
    total_to_compute = len(tasks)
    last_summary_write = time.perf_counter()
    while completed < total_to_compute:
        try:
            result = result_queue.get(timeout=5.0)
        except queue.Empty:
            alive = [proc.is_alive() for proc in workers]
            if not any(alive) and completed < total_to_compute:
                raise RuntimeError("All workers exited before completing the queued tasks")
            continue

        completed += 1
        if result["type"] == "success":
            aggregate_hit1_entries += int(result["hit1_entries"])
            aggregate_total_entries += int(result["total_entries_edge"])
            aggregate_occupied_pixels += int(result["occupied_pixels"])
            aggregate_single_hit_pixels += int(result["single_hit_pixels"])
            aggregate_multi_hit_pixels += int(result["multi_hit_pixels"])
            results_by_index[result["index"]] = {
                "uid": result["uid"],
                "tensor_path": result["tensor_path"],
                "storage_format_version": result["storage_format_version"],
                "model_component_shapes": result["model_component_shapes"],
                "model_rgb_shape": result["model_rgb_shape"],
                "model_depth_shape": result["model_depth_shape"],
                "model_normal_bytes_shape": result["model_normal_bytes_shape"],
                "edge_shape": result["edge_shape"],
                "edge_depth_shape": result["edge_depth_shape"],
                "storage_dtypes": result["storage_dtypes"],
                "model_storage_dtypes": result["model_storage_dtypes"],
                "edge_storage_dtypes": result["edge_storage_dtypes"],
                "hit1_entries": result["hit1_entries"],
                "total_entries_edge": result["total_entries_edge"],
                "hit1_over_total_entries_ratio": result["hit1_over_total_entries_ratio"],
                "occupied_pixels": result["occupied_pixels"],
                "single_hit_pixels": result["single_hit_pixels"],
                "multi_hit_pixels": result["multi_hit_pixels"],
                "single_hit_pixel_ratio": result["single_hit_pixel_ratio"],
                "multi_hit_pixel_ratio": result["multi_hit_pixel_ratio"],
                "elapsed_sec": result["elapsed_sec"],
                "source": result["source"],
                "worker": result["worker"],
                "device": result["device"],
            }
            print(
                f"[{completed:05d}/{total_to_compute:05d}] {result['uid']} saved on {result['device']} by {result['worker']} "
                f"hit1/edge={result['hit1_entries']}/{result['total_entries_edge']} ({result['hit1_over_total_entries_ratio']:.4f})"
            )
        else:
            failed_items.append(
                {
                    "uid": result["uid"],
                    "worker": result["worker"],
                    "device": result["device"],
                    "error_type": result["error_type"],
                    "error": result["error"],
                }
            )
            print(
                f"[{completed:05d}/{total_to_compute:05d}] {result['uid']} failed on {result['device']} by {result['worker']}, skipping: "
                f"{result['error_type']}: {result['error']}"
            )

        now = time.perf_counter()
        if completed % max(args.summary_every, 1) == 0 or now - last_summary_write > 30:
            _write_summary(
                summary_path,
                dataset_root=dataset_root,
                output_dir=output_dir,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
                resolution=args.resolution,
                model_max_hits=args.model_max_hits,
                edge_max_hits=args.edge_max_hits,
                device=",".join(worker_devices),
                model_backend=resolved_model_backend,
                reused_existing=reused_existing,
                failed_items=failed_items,
                aggregate_hit1_entries=aggregate_hit1_entries,
                aggregate_total_entries=aggregate_total_entries,
                aggregate_occupied_pixels=aggregate_occupied_pixels,
                aggregate_single_hit_pixels=aggregate_single_hit_pixels,
                aggregate_multi_hit_pixels=aggregate_multi_hit_pixels,
                per_uid_stats=[results_by_index[i] for i in sorted(results_by_index)],
            )
            last_summary_write = now

    for proc in workers:
        proc.join()

    _write_summary(
        summary_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        resolution=args.resolution,
        model_max_hits=args.model_max_hits,
        edge_max_hits=args.edge_max_hits,
        device=",".join(worker_devices),
        model_backend=resolved_model_backend,
        reused_existing=reused_existing,
        failed_items=failed_items,
        aggregate_hit1_entries=aggregate_hit1_entries,
        aggregate_total_entries=aggregate_total_entries,
        aggregate_occupied_pixels=aggregate_occupied_pixels,
        aggregate_single_hit_pixels=aggregate_single_hit_pixels,
        aggregate_multi_hit_pixels=aggregate_multi_hit_pixels,
        per_uid_stats=[results_by_index[i] for i in sorted(results_by_index)],
    )
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()