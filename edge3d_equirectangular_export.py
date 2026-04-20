import argparse
import csv
import json
from pathlib import Path
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from PIL import Image
import torch

from edge3d_pipeline import (
    DEFAULT_EDGE3D_ALIGNMENT,
    MeshCanonicalizer,
    ObjaverseEdgeDataset,
    ObjaverseModelProvider,
    load_alignment_from_report,
)
from utils.condition_metadata import condition_variant_specs
from inverse_spherical_representation import (
    InverseSphericalRepresentation,
    mesh_to_inverse_spherical_representation,
    polylines_to_inverse_spherical_representation,
)


FORMAT_VERSION = 2
CONDITION_VARIANTS = condition_variant_specs()


def _condition_variant_metadata() -> dict[str, dict[str, object]]:
    return {
        spec["label"]: {
            **spec["flags"],
            "legacy_condition_type_id": spec["legacy_condition_type_id"],
        }
        for spec in CONDITION_VARIANTS
    }


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _tile_layer_images(images: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(images, axis=1)


def _rgb_strip(rep: InverseSphericalRepresentation) -> np.ndarray:
    colors = rep.colors.cpu().numpy()
    valid = rep.valid_mask.cpu().numpy()
    layer_images = []
    for layer in range(colors.shape[0]):
        image = np.transpose(colors[layer], (1, 2, 0))
        image = np.clip(image, 0.0, 1.0)
        image[~valid[layer]] = 0.0
        layer_images.append((image * 255.0).astype(np.uint8))
    return _tile_layer_images(layer_images)


def _depth_layer_images(rep: InverseSphericalRepresentation) -> list[np.ndarray]:
    depth = rep.radii.cpu().numpy()
    valid = rep.valid_mask.cpu().numpy()
    layer_images = []
    if np.any(valid):
        depth_min = float(depth[valid].min())
        depth_max = float(depth[valid].max())
    else:
        depth_min = 0.0
        depth_max = 1.0
    denom = max(depth_max - depth_min, 1e-8)
    for layer in range(depth.shape[0]):
        image = np.zeros((depth.shape[1], depth.shape[2], 3), dtype=np.uint8)
        normalized = np.zeros_like(depth[layer], dtype=np.float32)
        normalized[valid[layer]] = 1.0 - np.clip((depth[layer][valid[layer]] - depth_min) / denom, 0.0, 1.0)
        gray = (normalized * 255.0).astype(np.uint8)
        image[valid[layer]] = np.stack([gray, gray, gray], axis=-1)[valid[layer]]
        layer_images.append(image)
    return layer_images


def _depth_strip(rep: InverseSphericalRepresentation) -> np.ndarray:
    return _tile_layer_images(_depth_layer_images(rep))


def _first_hit_depth_image(rep: InverseSphericalRepresentation) -> np.ndarray:
    return _depth_layer_images(rep)[0]


def _resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_backend(requested_backend: str, device: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    return "gpu_exact" if str(device).startswith("cuda") else "cpu_exact"


def _manifest_row_from_metadata(metadata: dict[str, object]) -> dict[str, object]:
    model_row = metadata["model"]
    edge_row = metadata["edge"]
    timing = metadata.get("timing_sec", {})
    return {
        "uid": metadata["uid"],
        "resolution": metadata["resolution"],
        "max_hits": metadata["max_hits"],
        "device": metadata.get("device", "cpu"),
        "model_backend": metadata.get("model_backend", "cpu_exact"),
        "model_rgb_path": model_row.get("rgb_path", ""),
        "model_depth_path": model_row["depth_path"],
        "model_depth_layers_path": model_row.get("depth_layers_path", ""),
        "model_normal_path": model_row["normal_path"],
        "model_representation_path": model_row["representation_path"],
        "edge_depth_path": edge_row["depth_path"],
        "edge_depth_layers_path": edge_row.get("depth_layers_path", ""),
        "edge_representation_path": edge_row["representation_path"],
        "model_prep_time_sec": timing.get("model_preparation", 0.0),
        "edge_prep_time_sec": timing.get("edge_preparation", 0.0),
        "asset_export_time_sec": timing.get("asset_export", 0.0),
        "total_time_sec": timing.get("total", 0.0),
    }


def _normal_strip(rep: InverseSphericalRepresentation) -> np.ndarray:
    if rep.normals is None:
        raise ValueError("Normal strip requested but representation has no normals")
    normals = rep.normals.cpu().numpy()
    valid = rep.valid_mask.cpu().numpy()
    layer_images = []
    for layer in range(normals.shape[0]):
        image = np.transpose(normals[layer], (1, 2, 0))
        image = np.clip((image + 1.0) * 0.5, 0.0, 1.0)
        image[~valid[layer]] = 0.0
        layer_images.append((image * 255.0).astype(np.uint8))
    return _tile_layer_images(layer_images)


def _build_training_representation(
    rep: InverseSphericalRepresentation,
    include_rgb: bool,
    include_normal: bool,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    valid = rep.valid_mask.cpu().numpy().astype(np.float32)
    depth = rep.radii.cpu().numpy().astype(np.float32)[:, None, :, :] * valid[:, None, :, :]
    payload = {
        "depth": depth,
        "valid_mask": valid.astype(np.uint8),
    }
    blocks = []
    channel_names: list[str] = []
    rgb = None
    if include_rgb:
        rgb = rep.colors.cpu().numpy().astype(np.float32) * valid[:, None, :, :]
        payload["rgb"] = rgb
    if include_normal:
        if rep.normals is None:
            raise ValueError("Normals are required but missing from representation")
        normal = rep.normals.cpu().numpy().astype(np.float32) * valid[:, None, :, :]
        payload["normal"] = normal
    else:
        normal = None

    for layer in range(depth.shape[0]):
        if include_rgb and rgb is not None:
            blocks.append(rgb[layer])
            channel_names.extend([f"hit{layer}_rgb_r", f"hit{layer}_rgb_g", f"hit{layer}_rgb_b"])
        blocks.append(depth[layer])
        channel_names.append(f"hit{layer}_depth")
        if include_normal and normal is not None:
            blocks.append(normal[layer])
            channel_names.extend([f"hit{layer}_normal_x", f"hit{layer}_normal_y", f"hit{layer}_normal_z"])

    representation = np.concatenate(blocks, axis=0).astype(np.float32)
    return representation, channel_names, payload


def _save_training_file(
    path: Path,
    representation: np.ndarray,
    channel_names: list[str],
    payload: dict[str, np.ndarray],
    uid: str,
    kind: str,
    resolution: int,
    max_hits: int,
) -> None:
    np.savez_compressed(
        path,
        uid=np.asarray(uid),
        kind=np.asarray(kind),
        resolution=np.asarray(resolution, dtype=np.int32),
        max_hits=np.asarray(max_hits, dtype=np.int32),
        channel_layout=np.asarray("per_hit_interleaved"),
        channel_names=np.asarray(channel_names),
        representation=representation,
        **payload,
    )


def _export_representation_assets(
    uid: str,
    kind: str,
    rep: InverseSphericalRepresentation,
    out_dir: Path,
    include_rgb_channels: bool,
    include_normal: bool,
    resolution: int,
    max_hits: int,
    save_rgb_visualization: bool = True,
    save_depth_layers_visualization: bool = False,
) -> dict[str, object]:
    rgb_file = out_dir / f"{kind}_rgb.png"
    rgb_path = None
    depth_path = out_dir / f"{kind}_depth.png"
    depth_layers_file = out_dir / f"{kind}_depth_layers.png"
    depth_layers_path = None

    if save_rgb_visualization:
        rgb_path = rgb_file
        _save_image(rgb_path, _rgb_strip(rep))
    elif rgb_file.exists():
        rgb_file.unlink()
    if save_depth_layers_visualization:
        _save_image(depth_path, _first_hit_depth_image(rep))
        depth_layers_path = depth_layers_file
        _save_image(depth_layers_path, _depth_strip(rep))
    else:
        _save_image(depth_path, _depth_strip(rep))
        if depth_layers_file.exists():
            depth_layers_file.unlink()

    normal_path = None
    if include_normal:
        normal_path = out_dir / f"{kind}_normal.png"
        _save_image(normal_path, _normal_strip(rep))

    representation, channel_names, payload = _build_training_representation(
        rep,
        include_rgb=include_rgb_channels,
        include_normal=include_normal,
    )
    train_path = out_dir / f"{kind}_representation.npz"
    _save_training_file(
        train_path,
        representation,
        channel_names,
        payload,
        uid=uid,
        kind=kind,
        resolution=resolution,
        max_hits=max_hits,
    )

    return {
        "uid": uid,
        "kind": kind,
        "rgb_path": str(rgb_path) if rgb_path is not None else "",
        "depth_path": str(depth_path),
        "depth_layers_path": str(depth_layers_path) if depth_layers_path is not None else "",
        "normal_path": str(normal_path) if normal_path is not None else "",
        "representation_path": str(train_path),
        "channel_count": int(representation.shape[0]),
        "height": int(representation.shape[1]),
        "width": int(representation.shape[2]),
    }


def _is_sample_complete(rep_dir: Path, resolution: int, max_hits: int, model_backend: str) -> bool:
    metadata_path = rep_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
        return False
    if int(metadata.get("resolution", -1)) != int(resolution):
        return False
    if int(metadata.get("max_hits", -1)) != int(max_hits):
        return False
    if metadata.get("model_backend") != model_backend:
        return False
    if metadata.get("edge_mode") != "polyline_single_pixel_rasterized":
        return False
    if metadata.get("edge_representation_mode") != "depth_only":
        return False
    required_files = [
        rep_dir / "model_rgb.png",
        rep_dir / "model_depth.png",
        rep_dir / "model_normal.png",
        rep_dir / "model_representation.npz",
        rep_dir / "edge_depth.png",
        rep_dir / "edge_depth_layers.png",
        rep_dir / "edge_representation.npz",
    ]
    return all(path.exists() for path in required_files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate equirectangular spherical representations for the 20 Edge3D study samples.")
    parser.add_argument("--dataset-root", default="/home/devdata/edge3d_data")
    parser.add_argument("--output-dir", default="demo_outputs/edge3d_20_sample_study")
    parser.add_argument("--selected-uids", default=None, help="Optional selected_uids.json override")
    parser.add_argument("--alignment-report", default=None, help="Optional alignment_report.json override")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--max-hits", type=int, default=5)
    parser.add_argument("--outer-radius", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--edge-color", default="0,190,255", help="Edge RGB color as R,G,B")
    parser.add_argument("--edge-sample-factor", type=float, default=2.0)
    parser.add_argument("--edge-depth-merge-tol", type=float, default=None)
    parser.add_argument("--download-processes", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-backend", default="auto", choices=["auto", "cpu_exact", "gpu_exact"])
    parser.add_argument("--shading", default="headlight", choices=["headlight", "none"])
    return parser.parse_args()


def _parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid RGB value: {value}")
    color = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError(f"RGB channel out of range: {value}")
    return color


def _load_or_download_mesh(model_provider: ObjaverseModelProvider, uid: str):
    try:
        return model_provider.load_mesh(uid)
    except FileNotFoundError:
        print(f"  mesh missing for {uid}, downloading on demand")
        model_provider.ensure_downloaded([uid])
        return model_provider.load_mesh(uid)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    selected_uids_path = Path(args.selected_uids) if args.selected_uids else output_dir / "selected_uids.json"
    alignment_report_path = Path(args.alignment_report) if args.alignment_report else output_dir / "alignment_report.json"

    selected_payload = json.loads(selected_uids_path.read_text(encoding="utf-8"))
    selected_uids = [str(uid) for uid in selected_payload["uids"]]
    alignment = load_alignment_from_report(alignment_report_path) if alignment_report_path.exists() else DEFAULT_EDGE3D_ALIGNMENT
    edge_color = _parse_rgb(args.edge_color)
    resolved_device = _resolve_device(args.device)
    resolved_model_backend = _resolve_model_backend(args.model_backend, resolved_device)

    if resolved_model_backend == "gpu_exact" and not str(resolved_device).startswith("cuda"):
        raise ValueError("gpu_exact backend requires a CUDA device")

    dataset = ObjaverseEdgeDataset(dataset_root)
    model_provider = ObjaverseModelProvider(dataset_root / "objaverse_cache", download_processes=args.download_processes)
    canonicalizer = MeshCanonicalizer(alignment)

    manifest_rows: list[dict[str, object]] = []
    for index, uid in enumerate(selected_uids, start=1):
        sample_start = time.perf_counter()
        rep_dir = output_dir / "samples" / uid / "equirectangular"
        rep_dir.mkdir(parents=True, exist_ok=True)
        if _is_sample_complete(rep_dir, resolution=args.resolution, max_hits=args.max_hits, model_backend=resolved_model_backend):
            metadata = json.loads((rep_dir / "metadata.json").read_text(encoding="utf-8"))
            manifest_rows.append(_manifest_row_from_metadata(metadata))
            print(f"[{index:02d}/{len(selected_uids)}] {uid} already complete, skipping")
            continue

        mesh = _load_or_download_mesh(model_provider, uid)
        canonical_mesh = canonicalizer.canonicalize_mesh(mesh)
        edge_polylines = dataset.load_edge_polylines(uid).astype(np.float32)
        edge_polylines = canonicalizer.canonicalize_polylines(edge_polylines)

        model_prep_start = time.perf_counter()
        model_rep = mesh_to_inverse_spherical_representation(
            canonical_mesh,
            resolution=args.resolution,
            max_hits=args.max_hits,
            outer_radius=args.outer_radius,
            batch_size=args.batch_size,
            shading=args.shading,
            device=resolved_device,
            stop_at_origin=True,
            backend=resolved_model_backend,
        )
        model_prep_time = round(time.perf_counter() - model_prep_start, 3)

        edge_prep_start = time.perf_counter()
        edge_rep = polylines_to_inverse_spherical_representation(
            edge_polylines,
            resolution=args.resolution,
            max_hits=args.max_hits,
            edge_color=edge_color,
            sample_factor=args.edge_sample_factor,
            depth_merge_tol=args.edge_depth_merge_tol,
            device=resolved_device,
        )
        edge_prep_time = round(time.perf_counter() - edge_prep_start, 3)

        export_start = time.perf_counter()

        model_row = _export_representation_assets(
            uid=uid,
            kind="model",
            rep=model_rep,
            out_dir=rep_dir,
            include_rgb_channels=True,
            include_normal=True,
            resolution=args.resolution,
            max_hits=args.max_hits,
            save_rgb_visualization=True,
            save_depth_layers_visualization=False,
        )
        edge_row = _export_representation_assets(
            uid=uid,
            kind="edge",
            rep=edge_rep,
            out_dir=rep_dir,
            include_rgb_channels=False,
            include_normal=False,
            resolution=args.resolution,
            max_hits=args.max_hits,
            save_rgb_visualization=False,
            save_depth_layers_visualization=True,
        )
        export_time = round(time.perf_counter() - export_start, 3)
        total_time = round(time.perf_counter() - sample_start, 3)

        metadata = {
            "format_version": FORMAT_VERSION,
            "uid": uid,
            "resolution": args.resolution,
            "max_hits": args.max_hits,
            "shading": args.shading,
            "device": resolved_device,
            "model_backend": resolved_model_backend,
            "edge_mode": "polyline_single_pixel_rasterized",
            "edge_representation_mode": "depth_only",
            "timing_sec": {
                "model_preparation": model_prep_time,
                "edge_preparation": edge_prep_time,
                "asset_export": export_time,
                "total": total_time,
            },
            "alignment": {
                "axis_order": list(alignment.axis_order),
                "axis_signs": list(alignment.axis_signs),
                "normalization": alignment.normalization,
            },
            "condition_variants": _condition_variant_metadata(),
            "model": model_row,
            "edge": edge_row,
        }
        (rep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        manifest_rows.append(_manifest_row_from_metadata(metadata))

        print(
            f"[{index:02d}/{len(selected_uids)}] {uid} exported equirectangular representation "
            f"backend={resolved_model_backend} model_prep={metadata['timing_sec']['model_preparation']:.3f}s "
            f"edge_prep={metadata['timing_sec']['edge_preparation']:.3f}s export={metadata['timing_sec']['asset_export']:.3f}s total={metadata['timing_sec']['total']:.3f}s"
        )

    manifest_path = output_dir / "equirectangular_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()