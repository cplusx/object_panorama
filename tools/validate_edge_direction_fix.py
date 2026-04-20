from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge3d_pipeline import (  # noqa: E402
    DEFAULT_EDGE3D_ALIGNMENT,
    MeshCanonicalizer,
    ObjaverseEdgeDataset,
    ObjaverseModelProvider,
    load_alignment_from_report,
)
from inverse_spherical_representation import (  # noqa: E402
    mesh_to_inverse_spherical_representation,
    polylines_to_inverse_spherical_representation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Edge3D edge-direction fix on a single UID.")
    parser.add_argument("--dataset-root", default="/home/devdata/edge3d_data")
    parser.add_argument("--uid", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alignment-report", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model-max-hits", type=int, default=5)
    parser.add_argument("--edge-max-hits", type=int, default=3)
    parser.add_argument("--outer-radius", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-backend", default="gpu_exact")
    parser.add_argument("--edge-sample-factor", type=float, default=2.0)
    parser.add_argument("--edge-depth-merge-tol", type=float, default=None)
    parser.add_argument("--download-processes", type=int, default=4)
    return parser.parse_args()


def load_or_download_mesh(model_provider: ObjaverseModelProvider, uid: str):
    try:
        return model_provider.load_mesh(uid)
    except FileNotFoundError:
        model_provider.ensure_downloaded([uid])
        return model_provider.load_mesh(uid)


def percentile_range(values: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lower, upper])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def make_depth_preview(depth_hit: np.ndarray, cmap_name: str) -> np.ndarray:
    lo, hi = percentile_range(depth_hit)
    safe = np.nan_to_num(depth_hit, nan=lo, posinf=hi, neginf=lo).astype(np.float32)
    normalized = np.clip((safe - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    image = plt.get_cmap(cmap_name)(normalized)[..., :3]
    image = image.copy()
    image[~np.isfinite(depth_hit)] = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    return image


def make_mask_preview(mask: np.ndarray) -> np.ndarray:
    image = np.zeros(mask.shape + (3,), dtype=np.float32)
    image[mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return image


def make_overlay(model_mask: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    image = np.zeros(model_mask.shape + (3,), dtype=np.float32)
    image[model_mask] = np.array([0.1, 0.8, 0.2], dtype=np.float32)
    image[edge_mask] = np.array([0.95, 0.15, 0.15], dtype=np.float32)
    image[np.logical_and(model_mask, edge_mask)] = np.array([1.0, 0.9, 0.15], dtype=np.float32)
    return image


def format_array(values: np.ndarray) -> str:
    return np.array2string(np.asarray(values, dtype=np.float32), precision=6, suppress_small=False)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alignment = load_alignment_from_report(args.alignment_report) if args.alignment_report else DEFAULT_EDGE3D_ALIGNMENT
    canonicalizer = MeshCanonicalizer(alignment)
    dataset = ObjaverseEdgeDataset(dataset_root)
    model_provider = ObjaverseModelProvider(dataset_root / "objaverse_cache", download_processes=args.download_processes)

    mesh = load_or_download_mesh(model_provider, args.uid)
    canonical_mesh = canonicalizer.canonicalize_mesh(mesh)
    edge_polylines = dataset.load_edge_polylines(args.uid).astype(np.float32)
    edge_points = edge_polylines.reshape(-1, 3)

    canonical_mesh_bbox = canonical_mesh.bounds.astype(np.float32)
    raw_edge_bbox = np.stack([edge_points.min(axis=0), edge_points.max(axis=0)], axis=0).astype(np.float32)
    canonical_mesh_extent = (canonical_mesh_bbox[1] - canonical_mesh_bbox[0]).astype(np.float32)
    raw_edge_extent = (raw_edge_bbox[1] - raw_edge_bbox[0]).astype(np.float32)
    canonical_mesh_radius_p99 = float(np.percentile(np.linalg.norm(np.asarray(canonical_mesh.vertices), axis=1), 99))
    raw_edge_radius_p99 = float(np.percentile(np.linalg.norm(edge_points, axis=1), 99))

    print("canonical mesh bbox:", format_array(canonical_mesh_bbox))
    print("raw edge bbox:", format_array(raw_edge_bbox))
    print("canonical mesh extent:", format_array(canonical_mesh_extent))
    print("raw edge extent:", format_array(raw_edge_extent))
    print("canonical mesh radius p99:", canonical_mesh_radius_p99)
    print("raw edge radius p99:", raw_edge_radius_p99)

    model_rep = mesh_to_inverse_spherical_representation(
        canonical_mesh,
        resolution=args.resolution,
        max_hits=args.model_max_hits,
        outer_radius=args.outer_radius,
        batch_size=args.batch_size,
        shading="headlight",
        device=args.device,
        stop_at_origin=True,
        backend=args.model_backend,
    )
    edge_rep = polylines_to_inverse_spherical_representation(
        edge_polylines,
        resolution=args.resolution,
        max_hits=args.edge_max_hits,
        edge_color=(0, 190, 255),
        sample_factor=args.edge_sample_factor,
        depth_merge_tol=args.edge_depth_merge_tol,
        device=args.device,
    )

    model_depth_hit1 = model_rep.radii[0].cpu().numpy()
    edge_depth_hit1 = edge_rep.radii[0].cpu().numpy()
    model_mask = np.any(model_rep.valid_mask.cpu().numpy(), axis=0)
    edge_mask = np.any(edge_rep.valid_mask.cpu().numpy(), axis=0)
    intersection = int(np.logical_and(model_mask, edge_mask).sum())
    union = int(np.logical_or(model_mask, edge_mask).sum())
    edge_pixels = int(edge_mask.sum())
    model_pixels = int(model_mask.sum())
    iou = float(intersection / max(union, 1))
    edge_overlap_ratio = float(intersection / max(edge_pixels, 1))
    model_overlap_ratio = float(intersection / max(model_pixels, 1))

    print("model pixels:", model_pixels)
    print("edge pixels:", edge_pixels)
    print("mask intersection:", intersection)
    print("mask union:", union)
    print("mask iou:", iou)
    print("edge overlap ratio:", edge_overlap_ratio)
    print("model overlap ratio:", model_overlap_ratio)

    figure, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    panels = [
        (make_depth_preview(model_depth_hit1, "magma"), "model depth hit1"),
        (make_depth_preview(edge_depth_hit1, "viridis"), "edge depth hit1"),
        (make_overlay(model_mask, edge_mask), f"overlay\nIoU={iou:.4f}"),
        (make_mask_preview(model_mask), "model any-hit mask"),
        (make_mask_preview(edge_mask), f"edge any-hit mask\nedge overlap={edge_overlap_ratio:.4f}"),
        (make_overlay(model_mask, edge_mask), f"overlap\nmodel overlap={model_overlap_ratio:.4f}"),
    ]
    for axis, (image, title) in zip(axes.flatten(), panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(f"Edge direction validation for {args.uid}", fontsize=16)
    figure_path = output_dir / f"{args.uid}_equirectangular_alignment_check.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    metrics_path = output_dir / f"{args.uid}_alignment_metrics.txt"
    metrics_path.write_text(
        "\n".join(
            [
                f"uid: {args.uid}",
                f"canonical mesh bbox: {format_array(canonical_mesh_bbox)}",
                f"raw edge bbox: {format_array(raw_edge_bbox)}",
                f"canonical mesh extent: {format_array(canonical_mesh_extent)}",
                f"raw edge extent: {format_array(raw_edge_extent)}",
                f"canonical mesh radius p99: {canonical_mesh_radius_p99}",
                f"raw edge radius p99: {raw_edge_radius_p99}",
                f"model pixels: {model_pixels}",
                f"edge pixels: {edge_pixels}",
                f"mask intersection: {intersection}",
                f"mask union: {union}",
                f"mask iou: {iou}",
                f"edge overlap ratio: {edge_overlap_ratio}",
                f"model overlap ratio: {model_overlap_ratio}",
                f"visualization: {figure_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("visualization:", figure_path)
    print("metrics:", metrics_path)


if __name__ == "__main__":
    main()