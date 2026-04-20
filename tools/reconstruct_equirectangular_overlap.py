from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge3d_tensor_format import load_sample_modalities
from reconstruction import (
    EDGE_HIT_COLOR_PALETTE_RGB,
    build_direction_map,
    decode_edge_points,
    export_overlap_pointcloud_glb,
    export_point_cloud,
)

try:
    from edge3d_pipeline import (
        DEFAULT_EDGE3D_ALIGNMENT,
        AxisAlignment,
        MeshCanonicalizer,
        ObjaverseEdgeDataset,
        ObjaverseModelProvider,
        load_alignment_from_report,
    )

    _HAS_REFERENCE_STACK = True
except ModuleNotFoundError:
    DEFAULT_EDGE3D_ALIGNMENT = None
    AxisAlignment = None
    MeshCanonicalizer = None
    ObjaverseEdgeDataset = None
    ObjaverseModelProvider = None
    load_alignment_from_report = None
    _HAS_REFERENCE_STACK = False

REFERENCE_POINT_CLOUD_RGB = np.array([0.72, 0.72, 0.72], dtype=np.float32)


@dataclass
class ReconstructionMetrics:
    count_reference: int
    count_reconstruction: int
    chamfer_mean: float
    chamfer_p90: float
    precision_2pct: float
    recall_2pct: float
    fscore_2pct: float


@dataclass
class ReconstructionStats:
    occupied_bins: int
    occupied_ratio: float
    multilayer_bins: int
    multilayer_ratio: float
    max_layer_observed: int
    reconstructed_points: int


@dataclass
class ReconstructionResult:
    uid: str
    npz_path: str
    overlap_pointcloud_glb_path: str
    reference_model_pointcloud_path: str | None
    reference_vs_reconstructed_pointcloud_path: str | None
    model_pointcloud_path: str
    edge_pointcloud_path: str
    model_points: int
    edge_points: int
    resolution: int
    edge_hit_palette_rgb: list[list[int]]
    edge_point_counts_per_hit: list[int]
    pointcloud_export_error: str | None
    model_stats: dict[str, Any]
    edge_stats: dict[str, Any]
    reference_metrics: dict[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct 3D overlap scenes from Edge3D equirectangular NPZ samples.")
    parser.add_argument("--input-path", default=None, help="Single NPZ file to reconstruct.")
    parser.add_argument("--input-dir", default=None, help="Directory of NPZ files to reconstruct.")
    parser.add_argument("--output-dir", required=True, help="Directory where reconstructed outputs will be written.")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index when scanning --input-dir.")
    parser.add_argument("--count", type=int, default=None, help="Optional number of files to process from --input-dir.")
    parser.add_argument("--num-process", type=int, default=1, help="Number of local worker processes for directory mode.")
    parser.add_argument("--dataset-root", default="/home/devdata/edge3d_data", help="Dataset root used for optional reference metrics.")
    parser.add_argument("--alignment-report", default=None, help="Optional alignment report used for reference mesh canonicalization.")
    parser.add_argument("--download-processes", type=int, default=4, help="Objaverse download workers when reference metrics are enabled.")
    parser.add_argument("--surface-sample-count", type=int, default=50000, help="Reference mesh samples used for reconstruction metrics.")
    parser.add_argument("--metric-threshold", type=float, default=0.02, help="Distance threshold used for precision/recall/F-score metrics.")
    parser.add_argument("--skip-reference-metrics", action="store_true", help="Skip comparison against original mesh and raw edge points.")
    return parser.parse_args()
def get_alignment(alignment_report: str | None):
    if alignment_report:
        return load_alignment_from_report(alignment_report)
    return DEFAULT_EDGE3D_ALIGNMENT


def compute_layer_stats(depth_layers: np.ndarray) -> ReconstructionStats:
    valid_mask = np.isfinite(depth_layers) & (depth_layers > 1e-8)
    layer_counts = valid_mask.sum(axis=0)
    occupied_bins = int(np.count_nonzero(layer_counts > 0))
    multilayer_bins = int(np.count_nonzero(layer_counts > 1))
    reconstructed_points = int(np.count_nonzero(valid_mask))
    height = int(depth_layers.shape[1])
    width = int(depth_layers.shape[2])
    return ReconstructionStats(
        occupied_bins=occupied_bins,
        occupied_ratio=float(occupied_bins / max(height * width, 1)),
        multilayer_bins=multilayer_bins,
        multilayer_ratio=float(multilayer_bins / max(occupied_bins, 1)),
        max_layer_observed=int(layer_counts.max()) if occupied_bins else 0,
        reconstructed_points=reconstructed_points,
    )


def decode_model_points_and_colors(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model_rgb = np.asarray(sample["model_rgb"], dtype=np.float32)
    model_depth = np.asarray(sample["model_depth"], dtype=np.float32)
    directions = build_direction_map(int(sample["resolution"]))

    point_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    for hit_index in range(model_depth.shape[0]):
        depth_hit = model_depth[hit_index]
        valid_mask = np.isfinite(depth_hit) & (depth_hit > 1e-8)
        if not np.any(valid_mask):
            continue
        points = directions[valid_mask] * depth_hit[valid_mask, None].astype(np.float32)
        colors = np.moveaxis(model_rgb[hit_index], 0, -1)[valid_mask].astype(np.float32)
        point_blocks.append(points.astype(np.float32))
        color_blocks.append(np.clip(colors, 0.0, 1.0).astype(np.float32))

    if not point_blocks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(point_blocks, axis=0), np.concatenate(color_blocks, axis=0)
def export_combined_point_cloud(
    point_sets: list[np.ndarray],
    color_sets: list[np.ndarray],
    output_path: Path,
) -> None:
    non_empty_points = [np.asarray(points, dtype=np.float32) for points in point_sets if len(points) > 0]
    non_empty_colors = [np.asarray(colors, dtype=np.float32) for points, colors in zip(point_sets, color_sets) if len(points) > 0]
    if not non_empty_points:
        export_point_cloud(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), output_path)
        return
    export_point_cloud(np.concatenate(non_empty_points, axis=0), np.concatenate(non_empty_colors, axis=0), output_path)


def load_or_download_mesh(model_provider: ObjaverseModelProvider, uid: str) -> trimesh.Trimesh:
    try:
        return model_provider.load_mesh(uid)
    except FileNotFoundError:
        model_provider.ensure_downloaded([uid])
        return model_provider.load_mesh(uid)


def compute_metrics(reference: np.ndarray, reconstruction: np.ndarray, threshold: float) -> ReconstructionMetrics:
    reference = np.asarray(reference, dtype=np.float32)
    reconstruction = np.asarray(reconstruction, dtype=np.float32)
    if len(reference) == 0 or len(reconstruction) == 0:
        return ReconstructionMetrics(
            count_reference=int(len(reference)),
            count_reconstruction=int(len(reconstruction)),
            chamfer_mean=float("inf"),
            chamfer_p90=float("inf"),
            precision_2pct=0.0,
            recall_2pct=0.0,
            fscore_2pct=0.0,
        )

    tree_ref = cKDTree(reference)
    tree_rec = cKDTree(reconstruction)
    dist_rec_ref, _ = tree_ref.query(reconstruction, k=1)
    dist_ref_rec, _ = tree_rec.query(reference, k=1)
    chamfer_mean = 0.5 * (dist_rec_ref.mean() + dist_ref_rec.mean())
    chamfer_p90 = 0.5 * (np.percentile(dist_rec_ref, 90) + np.percentile(dist_ref_rec, 90))
    precision = float((dist_rec_ref < threshold).mean())
    recall = float((dist_ref_rec < threshold).mean())
    fscore = float(2.0 * precision * recall / max(precision + recall, 1e-8))
    return ReconstructionMetrics(
        count_reference=int(len(reference)),
        count_reconstruction=int(len(reconstruction)),
        chamfer_mean=float(chamfer_mean),
        chamfer_p90=float(chamfer_p90),
        precision_2pct=precision,
        recall_2pct=recall,
        fscore_2pct=fscore,
    )


def can_compute_reference_metrics(dataset_root: Path, skip_reference_metrics: bool) -> bool:
    if skip_reference_metrics:
        return False
    if not _HAS_REFERENCE_STACK:
        return False
    return (dataset_root / "models").exists()


def compute_reference_metrics(
    canonical_mesh: trimesh.Trimesh,
    raw_edge_points: np.ndarray,
    surface_sample_count: int,
    metric_threshold: float,
    model_reconstruction: np.ndarray,
    edge_reconstruction: np.ndarray,
) -> dict[str, Any]:
    model_points = canonical_mesh.sample(int(surface_sample_count)).astype(np.float32)

    model_metrics = compute_metrics(model_points, model_reconstruction, threshold=metric_threshold)
    edge_metrics = compute_metrics(raw_edge_points, edge_reconstruction, threshold=metric_threshold)

    extents = (canonical_mesh.bounds[1] - canonical_mesh.bounds[0]).astype(np.float32)
    edge_bbox = np.stack([raw_edge_points.min(axis=0), raw_edge_points.max(axis=0)], axis=0).astype(np.float32)
    return {
        "bbox_extent_x": float(extents[0]),
        "bbox_extent_y": float(extents[1]),
        "bbox_extent_z": float(extents[2]),
        "bbox_anisotropy": float(np.max(extents) / max(np.min(extents), 1e-8)),
        "canonical_mesh_bbox": canonical_mesh.bounds.astype(np.float32).tolist(),
        "raw_edge_bbox": edge_bbox.tolist(),
        "model": asdict(model_metrics),
        "edge": asdict(edge_metrics),
    }


def reconstruct_equirectangular_npz_to_pointclouds(
    npz_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_root: str | Path = "/home/devdata/edge3d_data",
    alignment_report: str | None = None,
    download_processes: int = 4,
    surface_sample_count: int = 50000,
    metric_threshold: float = 0.02,
    skip_reference_metrics: bool = False,
) -> ReconstructionResult:
    npz_path = Path(npz_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_sample_modalities(npz_path)
    uid = str(sample["uid"])
    sample_dir = output_dir / uid
    sample_dir.mkdir(parents=True, exist_ok=True)

    model_points, model_colors = decode_model_points_and_colors(sample)
    edge_points, edge_colors, edge_point_counts_per_hit = decode_edge_points(
        np.asarray(sample["edge_depth"], dtype=np.float32),
        resolution=int(sample["resolution"]),
        color_palette_rgb=EDGE_HIT_COLOR_PALETTE_RGB,
    )

    overlap_pointcloud_glb_path = sample_dir / "overlap_pointcloud.glb"
    reference_model_pointcloud_path: Path | None = None
    reference_vs_reconstructed_pointcloud_path: Path | None = None
    model_pointcloud_path = sample_dir / "model_points.ply"
    edge_pointcloud_path = sample_dir / "edge_points.ply"

    export_point_cloud(model_points, model_colors, model_pointcloud_path)
    export_point_cloud(edge_points, edge_colors, edge_pointcloud_path)
    export_overlap_pointcloud_glb(
        model_points=model_points,
        model_colors=model_colors,
        edge_points=edge_points,
        edge_colors=edge_colors,
        output_path=overlap_pointcloud_glb_path,
    )

    model_stats = asdict(compute_layer_stats(np.asarray(sample["model_depth"], dtype=np.float32)))
    edge_stats = asdict(compute_layer_stats(np.asarray(sample["edge_depth"], dtype=np.float32)))

    reference_metrics = None
    pointcloud_export_error = None
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    canonical_reference_mesh = None
    raw_edge_points = None
    if _HAS_REFERENCE_STACK:
        try:
            alignment = get_alignment(alignment_report)
            canonicalizer = MeshCanonicalizer(alignment)
            model_provider = ObjaverseModelProvider(dataset_root_path / "objaverse_cache", download_processes=download_processes)
            raw_mesh = load_or_download_mesh(model_provider, uid)
            canonical_reference_mesh = canonicalizer.canonicalize_mesh(raw_mesh)
            reference_model_points = canonical_reference_mesh.sample(int(surface_sample_count)).astype(np.float32)
            reference_model_colors = np.repeat(REFERENCE_POINT_CLOUD_RGB[None, :], repeats=len(reference_model_points), axis=0)
            reference_model_pointcloud_path = sample_dir / f"{uid}_reference_model_canonical.ply"
            reference_vs_reconstructed_pointcloud_path = sample_dir / f"{uid}_reference_vs_reconstructed.ply"
            export_point_cloud(reference_model_points, reference_model_colors, reference_model_pointcloud_path)
            export_combined_point_cloud(
                point_sets=[reference_model_points, model_points, edge_points],
                color_sets=[reference_model_colors, model_colors, edge_colors],
                output_path=reference_vs_reconstructed_pointcloud_path,
            )
        except Exception as exc:
            pointcloud_export_error = f"{type(exc).__name__}: {exc}"

    if can_compute_reference_metrics(dataset_root_path, skip_reference_metrics):
        try:
            if canonical_reference_mesh is None:
                alignment = get_alignment(alignment_report)
                canonicalizer = MeshCanonicalizer(alignment)
                model_provider = ObjaverseModelProvider(dataset_root_path / "objaverse_cache", download_processes=download_processes)
                raw_mesh = load_or_download_mesh(model_provider, uid)
                canonical_reference_mesh = canonicalizer.canonicalize_mesh(raw_mesh)
            dataset = ObjaverseEdgeDataset(dataset_root_path)
            raw_edge_points = dataset.load_edge_points(uid).astype(np.float32)
            reference_metrics = compute_reference_metrics(
                canonical_mesh=canonical_reference_mesh,
                raw_edge_points=raw_edge_points,
                surface_sample_count=surface_sample_count,
                metric_threshold=metric_threshold,
                model_reconstruction=model_points,
                edge_reconstruction=edge_points,
            )
        except Exception as exc:
            reference_metrics = {"error": f"{type(exc).__name__}: {exc}"}

    result = ReconstructionResult(
        uid=uid,
        npz_path=str(npz_path),
        overlap_pointcloud_glb_path=str(overlap_pointcloud_glb_path),
        reference_model_pointcloud_path=str(reference_model_pointcloud_path) if reference_model_pointcloud_path is not None else None,
        reference_vs_reconstructed_pointcloud_path=str(reference_vs_reconstructed_pointcloud_path) if reference_vs_reconstructed_pointcloud_path is not None else None,
        model_pointcloud_path=str(model_pointcloud_path),
        edge_pointcloud_path=str(edge_pointcloud_path),
        model_points=int(len(model_points)),
        edge_points=int(len(edge_points)),
        resolution=int(sample["resolution"]),
        edge_hit_palette_rgb=[list(rgb) for rgb in EDGE_HIT_COLOR_PALETTE_RGB[: np.asarray(sample["edge_depth"]).shape[0]]],
        edge_point_counts_per_hit=edge_point_counts_per_hit,
        pointcloud_export_error=pointcloud_export_error,
        model_stats=model_stats,
        edge_stats=edge_stats,
        reference_metrics=reference_metrics,
    )

    report_path = sample_dir / f"{uid}_reconstruction_report.json"
    report_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


def reconstruct_equirectangular_npz_to_overlap_glb(
    npz_path: str | Path,
    output_dir: str | Path,
    *,
    dataset_root: str | Path = "/home/devdata/edge3d_data",
    alignment_report: str | None = None,
    download_processes: int = 4,
    surface_sample_count: int = 50000,
    metric_threshold: float = 0.02,
    skip_reference_metrics: bool = False,
) -> ReconstructionResult:
    return reconstruct_equirectangular_npz_to_pointclouds(
        npz_path=npz_path,
        output_dir=output_dir,
        dataset_root=dataset_root,
        alignment_report=alignment_report,
        download_processes=download_processes,
        surface_sample_count=surface_sample_count,
        metric_threshold=metric_threshold,
        skip_reference_metrics=skip_reference_metrics,
    )


def resolve_input_paths(input_path: str | None, input_dir: str | None, start_idx: int, count: int | None) -> list[Path]:
    if input_path:
        return [Path(input_path).expanduser().resolve()]
    if not input_dir:
        raise ValueError("Provide either --input-path or --input-dir")
    paths = sorted(Path(input_dir).expanduser().resolve().glob("*.npz"))
    sliced = paths[max(int(start_idx), 0) :]
    if count is not None:
        sliced = sliced[: max(int(count), 0)]
    return sliced


def process_path(task: dict[str, Any]) -> dict[str, Any]:
    result = reconstruct_equirectangular_npz_to_pointclouds(
        npz_path=task["npz_path"],
        output_dir=task["output_dir"],
        dataset_root=task["dataset_root"],
        alignment_report=task["alignment_report"],
        download_processes=int(task["download_processes"]),
        surface_sample_count=int(task["surface_sample_count"]),
        metric_threshold=float(task["metric_threshold"]),
        skip_reference_metrics=bool(task["skip_reference_metrics"]),
    )
    payload = asdict(result)
    payload["status"] = "ok"
    return payload


def run_tasks(tasks: list[dict[str, Any]], num_process: int) -> list[dict[str, Any]]:
    if int(num_process) <= 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            print(f"[{index:04d}/{len(tasks):04d}] reconstructing {task['npz_path']}")
            results.append(process_path(task))
        return results

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(num_process)) as executor:
        future_map = {executor.submit(process_path, task): task for task in tasks}
        for completed_index, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            print(f"[{completed_index:04d}/{len(tasks):04d}] finished {task['npz_path']}")
            results.append(future.result())
    results.sort(key=lambda item: str(item["npz_path"]))
    return results


def write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "reconstruction_summary.json"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if not results:
        return

    csv_rows = []
    for result in results:
        row = {
            "uid": result["uid"],
            "npz_path": result["npz_path"],
            "overlap_pointcloud_glb_path": result["overlap_pointcloud_glb_path"],
            "reference_model_pointcloud_path": result.get("reference_model_pointcloud_path"),
            "reference_vs_reconstructed_pointcloud_path": result.get("reference_vs_reconstructed_pointcloud_path"),
            "model_pointcloud_path": result["model_pointcloud_path"],
            "edge_pointcloud_path": result["edge_pointcloud_path"],
            "model_points": result["model_points"],
            "edge_points": result["edge_points"],
            "resolution": result["resolution"],
            "model_occupied_bins": result["model_stats"]["occupied_bins"],
            "model_multilayer_bins": result["model_stats"]["multilayer_bins"],
            "edge_occupied_bins": result["edge_stats"]["occupied_bins"],
            "edge_multilayer_bins": result["edge_stats"]["multilayer_bins"],
        }
        if result.get("pointcloud_export_error"):
            row["pointcloud_export_error"] = result["pointcloud_export_error"]
        reference_metrics = result.get("reference_metrics") or {}
        if isinstance(reference_metrics, dict) and "model" in reference_metrics and "edge" in reference_metrics:
            row.update(
                {
                    "model_chamfer_mean": reference_metrics["model"].get("chamfer_mean"),
                    "model_fscore_2pct": reference_metrics["model"].get("fscore_2pct"),
                    "edge_chamfer_mean": reference_metrics["edge"].get("chamfer_mean"),
                    "edge_fscore_2pct": reference_metrics["edge"].get("fscore_2pct"),
                }
            )
        elif isinstance(reference_metrics, dict) and "error" in reference_metrics:
            row["reference_metrics_error"] = reference_metrics["error"]
        csv_rows.append(row)

    summary_csv = output_dir / "reconstruction_summary.csv"
    fieldnames = []
    for row in csv_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def main() -> None:
    args = parse_args()
    sample_paths = resolve_input_paths(args.input_path, args.input_dir, args.start_idx, args.count)
    if not sample_paths:
        raise ValueError("No NPZ files selected")

    output_dir = Path(args.output_dir).expanduser().resolve()
    tasks = [
        {
            "npz_path": str(path),
            "output_dir": str(output_dir),
            "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
            "alignment_report": args.alignment_report,
            "download_processes": int(args.download_processes),
            "surface_sample_count": int(args.surface_sample_count),
            "metric_threshold": float(args.metric_threshold),
            "skip_reference_metrics": bool(args.skip_reference_metrics),
        }
        for path in sample_paths
    ]
    results = run_tasks(tasks, num_process=args.num_process)
    write_summary(output_dir, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()