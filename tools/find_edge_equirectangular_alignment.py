from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from edge3d.generation.pipeline import AxisAlignment, DEFAULT_EDGE3D_ALIGNMENT, MeshCanonicalizer, ObjaverseEdgeDataset
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class AxisAlignment:
        axis_order: tuple[int, int, int]
        axis_signs: tuple[int, int, int]
        normalization: str = "bbox"


    DEFAULT_EDGE3D_ALIGNMENT = AxisAlignment(axis_order=(0, 2, 1), axis_signs=(1, -1, 1))


    class ObjaverseEdgeDataset:
        def __init__(self, root_dir: str | Path):
            self.root_dir = Path(root_dir)
            self.model_dir = self.root_dir / "models"
            if not self.model_dir.exists():
                raise FileNotFoundError(self.model_dir)

        def load_edge_polylines(self, uid: str) -> np.ndarray:
            data = np.load(self.model_dir / f"{uid}.npz")
            features = data["features"].astype(np.float32)
            lower = data["lowerbounds"].astype(np.float32)
            upper = data["upperbounds"].astype(np.float32)
            return ((features + 1.0) * 0.5) * (upper - lower)[:, None, :] + lower[:, None, :]


    class MeshCanonicalizer:
        def __init__(self, alignment: AxisAlignment):
            self.alignment = alignment

        def normalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
            if self.alignment.normalization != "bbox":
                raise ValueError(f"Unsupported normalization {self.alignment.normalization}")
            min_v = vertices.min(axis=0)
            max_v = vertices.max(axis=0)
            center = 0.5 * (min_v + max_v)
            scale = float(np.max(max_v - min_v))
            scale = max(scale, 1e-8)
            return ((vertices - center) / scale * 2.0).astype(np.float32)

        def apply_axis_alignment(self, points: np.ndarray) -> np.ndarray:
            aligned = points[:, self.alignment.axis_order]
            signs = np.array(self.alignment.axis_signs, dtype=np.float32)
            return aligned * signs[None, :]

        def canonicalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
            return self.apply_axis_alignment(self.normalize_vertices(vertices))

        def canonicalize_points(self, points: np.ndarray) -> np.ndarray:
            points = np.asarray(points, dtype=np.float32)
            original_shape = points.shape
            if original_shape[-1] != 3:
                raise ValueError(f"Expected last dim = 3, got shape {original_shape}")
            flat = points.reshape(-1, 3)
            canonical = self.canonicalize_vertices(flat)
            return canonical.reshape(original_shape)

        def canonicalize_polylines(self, polylines: np.ndarray) -> np.ndarray:
            polylines = np.asarray(polylines, dtype=np.float32)
            if polylines.ndim != 3 or polylines.shape[-1] != 3:
                raise ValueError(f"Expected polylines shape [N, M, 3], got {polylines.shape}")
            return self.canonicalize_points(polylines)


def split_legacy_model_tensor(model_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_tensor = np.asarray(model_tensor, dtype=np.float32)
    model_rgb = np.stack([model_tensor[offset::7] for offset in (0, 1, 2)], axis=1)
    model_depth = model_tensor[3::7]
    model_normal = np.stack([model_tensor[offset::7] for offset in (4, 5, 6)], axis=1)
    return model_rgb.astype(np.float32), model_depth.astype(np.float32), model_normal.astype(np.float32)


def decode_fp8_e4m3fn_from_uint8(raw_values: np.ndarray) -> np.ndarray | None:
    if not hasattr(torch, "float8_e4m3fn"):
        return None
    tensor = torch.from_numpy(np.ascontiguousarray(raw_values.astype(np.uint8, copy=False)))
    return tensor.view(torch.float8_e4m3fn).to(torch.float32).cpu().numpy()


def load_sample_modalities_local(path: str | Path) -> dict[str, object]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        if "model_rgb" in payload.files and "model_depth" in payload.files and "edge_depth" in payload.files:
            model_normal = None
            if "model_normal_fp8_e4m3fn_bytes" in payload.files:
                model_normal = decode_fp8_e4m3fn_from_uint8(payload["model_normal_fp8_e4m3fn_bytes"])
            elif "model_normal" in payload.files:
                model_normal = payload["model_normal"].astype(np.float32)
            return {
                "uid": str(np.asarray(payload["uid"]).item()),
                "resolution": int(np.asarray(payload["resolution"]).item()),
                "model_rgb": payload["model_rgb"].astype(np.float32),
                "model_depth": payload["model_depth"].astype(np.float32),
                "model_normal": model_normal,
                "edge_depth": payload["edge_depth"].astype(np.float32),
            }

        model_tensor = payload["model_tensor"].astype(np.float32)
        edge_tensor = payload["edge_tensor"].astype(np.float32)
        model_rgb, model_depth, model_normal = split_legacy_model_tensor(model_tensor)
        return {
            "uid": str(np.asarray(payload["uid"]).item()),
            "resolution": int(model_tensor.shape[1]),
            "model_rgb": model_rgb,
            "model_depth": model_depth,
            "model_normal": model_normal,
            "edge_depth": edge_tensor,
        }


def project_points_to_equirectangular_pixels(points: np.ndarray, height: int) -> tuple[np.ndarray, np.ndarray]:
    width = 2 * height
    radii = np.linalg.norm(points, axis=1)
    keep = radii > 1e-8
    points = points[keep]
    radii = radii[keep]
    if len(points) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    directions = points / radii[:, None]
    phi = np.arccos(np.clip(directions[:, 1], -1.0, 1.0))
    theta = np.arctan2(directions[:, 0], directions[:, 2])
    rows = np.clip((phi / math.pi * height).astype(np.int32), 0, height - 1)
    cols = np.clip(((theta + math.pi) / (2.0 * math.pi) * width).astype(np.int32), 0, width - 1)
    return rows, cols


def wrapped_column_delta(start_col: int, end_col: int, width: int) -> int:
    delta = int(end_col) - int(start_col)
    half_width = width // 2
    if delta > half_width:
        delta -= width
    elif delta < -half_width:
        delta += width
    return delta


def rasterize_wrapped_pixel_line(
    start_row: int,
    start_col: int,
    end_row: int,
    end_col: int,
    height: int,
    width: int,
):
    delta_row = int(end_row) - int(start_row)
    delta_col = wrapped_column_delta(int(start_col), int(end_col), width)
    steps = max(abs(delta_row), abs(delta_col))
    if steps == 0:
        yield int(np.clip(start_row, 0, height - 1)), int(start_col) % width
        return

    last_key = None
    for step in range(steps + 1):
        t = step / steps
        row = int(np.clip(round(start_row + delta_row * t), 0, height - 1))
        col = int(round(start_col + delta_col * t)) % width
        key = (row, col)
        if key == last_key:
            continue
        yield key
        last_key = key


def project_polylines_to_occupancy(polylines: np.ndarray, resolution: int, sample_factor: float) -> np.ndarray:
    height = int(resolution)
    width = 2 * height
    occupancy = np.zeros((height, width), dtype=bool)
    sample_factor = float(sample_factor)

    for polyline in np.asarray(polylines, dtype=np.float32):
        if len(polyline) < 2:
            continue
        for start, end in zip(polyline[:-1], polyline[1:]):
            start_norm = float(np.linalg.norm(start))
            end_norm = float(np.linalg.norm(end))
            if start_norm <= 1e-8 or end_norm <= 1e-8:
                continue

            start_dir = start / start_norm
            end_dir = end / end_norm
            angular_distance = math.acos(float(np.clip(np.dot(start_dir, end_dir), -1.0, 1.0)))
            steps = max(2, int(math.ceil(angular_distance * height * sample_factor)) + 1)
            t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
            segment_points = start[None, :] * (1.0 - t) + end[None, :] * t
            rows, cols = project_points_to_equirectangular_pixels(segment_points, height)
            if len(rows) == 0:
                continue
            if len(rows) == 1:
                occupancy[int(rows[0]), int(cols[0])] = True
                continue

            for row0, col0, row1, col1 in zip(rows[:-1], cols[:-1], rows[1:], cols[1:]):
                for raster_row, raster_col in rasterize_wrapped_pixel_line(
                    int(row0),
                    int(col0),
                    int(row1),
                    int(col1),
                    height=height,
                    width=width,
                ):
                    occupancy[raster_row, raster_col] = True

    return occupancy


CURRENT_EDGE_EXTRA_ALIGNMENT = AxisAlignment(axis_order=(0, 1, 2), axis_signs=(1, 1, 1))


@dataclass(frozen=True)
class SearchStage:
    name: str
    resolution: int
    max_hits: int
    sample_factor: float
    depth_merge_tol: float | None


@dataclass(frozen=True)
class SampleArtifacts:
    uid: str
    sample_resolution: int
    score_maps: dict[str, np.ndarray]
    background_image: np.ndarray
    full_score_map: np.ndarray
    canonical_polylines: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search an edge-only axis transform that aligns edge equirectangular projections to model projections.")
    parser.add_argument("--dataset-root", default="/home/devdata/edge3d_data")
    parser.add_argument("--sample-dir", default="/home/devdata/edge3d_data/equirectangular_data")
    parser.add_argument("--output-dir", default="analysis/edge_orientation_search")
    parser.add_argument("--uids", default=None, help="Comma-separated list of sample UIDs. If omitted, use files from sample-dir.")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--reference-uid", default="15c420f2a56a419b9cff87573763c05c")
    parser.add_argument("--coarse-resolution", type=int, default=64)
    parser.add_argument("--fine-resolution", type=int, default=128)
    parser.add_argument("--coarse-max-hits", type=int, default=1)
    parser.add_argument("--fine-max-hits", type=int, default=3)
    parser.add_argument("--coarse-sample-factor", type=float, default=0.5)
    parser.add_argument("--fine-sample-factor", type=float, default=1.0)
    parser.add_argument("--refine-top-k", type=int, default=6)
    parser.add_argument("--visualize-top-k", type=int, default=4)
    parser.add_argument("--visualize-limit", type=int, default=3)
    return parser.parse_args()


def enumerate_candidates() -> list[AxisAlignment]:
    candidates: list[AxisAlignment] = []
    for axis_order in itertools.permutations((0, 1, 2)):
        for axis_signs in itertools.product((-1, 1), repeat=3):
            candidates.append(AxisAlignment(axis_order=axis_order, axis_signs=axis_signs))
    return candidates


def candidate_key(alignment: AxisAlignment) -> str:
    order = "".join(str(axis) for axis in alignment.axis_order)
    signs = "".join("p" if sign > 0 else "n" for sign in alignment.axis_signs)
    return f"order_{order}__signs_{signs}"


def candidate_label(alignment: AxisAlignment) -> str:
    return f"order={alignment.axis_order}, signs={alignment.axis_signs}"


def is_current_candidate(alignment: AxisAlignment) -> bool:
    return alignment.axis_order == CURRENT_EDGE_EXTRA_ALIGNMENT.axis_order and alignment.axis_signs == CURRENT_EDGE_EXTRA_ALIGNMENT.axis_signs


def resolve_uids(sample_dir: Path, explicit_uids: str | None, limit: int, reference_uid: str | None) -> list[str]:
    if explicit_uids:
        uids = [uid.strip() for uid in explicit_uids.split(",") if uid.strip()]
    else:
        uids = [path.stem for path in sorted(sample_dir.glob("*.npz"))]
    if reference_uid and reference_uid in uids:
        uids = [reference_uid] + [uid for uid in uids if uid != reference_uid]
    if limit > 0:
        uids = uids[:limit]
    return uids


def apply_axis_alignment(points: np.ndarray, alignment: AxisAlignment) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    original_shape = array.shape
    flat = array.reshape(-1, 3)
    aligned = flat[:, alignment.axis_order]
    signs = np.asarray(alignment.axis_signs, dtype=np.float32)
    aligned = aligned * signs[None, :]
    return aligned.reshape(original_shape)


def downsample_mean(image: np.ndarray, target_height: int) -> np.ndarray:
    source_height, source_width = image.shape
    target_width = target_height * 2
    if source_height == target_height and source_width == target_width:
        return image.astype(np.float32, copy=False)
    row_factor = source_height // target_height
    col_factor = source_width // target_width
    if source_height % target_height != 0 or source_width % target_width != 0:
        row_idx = np.linspace(0, source_height - 1, target_height).round().astype(np.int32)
        col_idx = np.linspace(0, source_width - 1, target_width).round().astype(np.int32)
        return image[row_idx][:, col_idx].astype(np.float32)
    reshaped = image.reshape(target_height, row_factor, target_width, col_factor)
    return reshaped.mean(axis=(1, 3), dtype=np.float32)


def normalize_map(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = float(np.percentile(finite, percentile))
    if scale <= 1e-8:
        scale = float(np.max(np.abs(finite)))
    if scale <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values.astype(np.float32) / scale, 0.0, 1.0)


def horizontal_gradient(values: np.ndarray) -> np.ndarray:
    return 0.5 * (np.abs(values - np.roll(values, 1, axis=1)) + np.abs(values - np.roll(values, -1, axis=1)))


def vertical_gradient(values: np.ndarray) -> np.ndarray:
    gradient = np.zeros_like(values, dtype=np.float32)
    gradient[1:] += np.abs(values[1:] - values[:-1])
    gradient[:-1] += np.abs(values[:-1] - values[1:])
    return 0.5 * gradient


def boundary_map(valid_mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(valid_mask.shape, dtype=np.float32)
    boundary += np.logical_xor(valid_mask, np.roll(valid_mask, 1, axis=1)).astype(np.float32)
    boundary += np.logical_xor(valid_mask, np.roll(valid_mask, -1, axis=1)).astype(np.float32)
    boundary[1:] += np.logical_xor(valid_mask[1:], valid_mask[:-1]).astype(np.float32)
    boundary[:-1] += np.logical_xor(valid_mask[:-1], valid_mask[1:]).astype(np.float32)
    return np.clip(boundary, 0.0, 1.0)


def build_target_score_map(sample: dict[str, np.ndarray]) -> np.ndarray:
    model_depth = np.asarray(sample["model_depth"], dtype=np.float32)
    model_normal = sample["model_normal"]
    score_map = np.zeros(model_depth.shape[1:], dtype=np.float32)

    for hit_index in range(model_depth.shape[0]):
        depth_hit = model_depth[hit_index]
        valid_mask = np.isfinite(depth_hit)
        if not np.any(valid_mask):
            continue

        safe_depth = np.nan_to_num(depth_hit, nan=0.0, posinf=0.0, neginf=0.0)
        depth_grad = horizontal_gradient(safe_depth) + vertical_gradient(safe_depth)
        depth_grad *= valid_mask.astype(np.float32)

        normal_grad = np.zeros_like(depth_hit, dtype=np.float32)
        if model_normal is not None:
            normal_hit = np.asarray(model_normal[hit_index], dtype=np.float32)
            for component in range(normal_hit.shape[0]):
                safe_component = np.nan_to_num(normal_hit[component], nan=0.0, posinf=0.0, neginf=0.0)
                normal_grad += horizontal_gradient(safe_component) + vertical_gradient(safe_component)
        normal_grad *= valid_mask.astype(np.float32)

        score_map += 1.5 * boundary_map(valid_mask)
        score_map += 1.0 * normalize_map(depth_grad)
        score_map += 0.75 * normalize_map(normal_grad)

    score_map = normalize_map(score_map, percentile=99.5)
    return score_map.astype(np.float32)


def percentile_range(values: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lower, upper])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def build_depth_background(depth_hit: np.ndarray) -> np.ndarray:
    lo, hi = percentile_range(depth_hit)
    safe = np.nan_to_num(depth_hit, nan=lo, posinf=hi, neginf=lo).astype(np.float32)
    normalized = np.clip((safe - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    background = np.stack([normalized, normalized, normalized], axis=-1)
    background[~np.isfinite(depth_hit)] = np.array([0.06, 0.06, 0.06], dtype=np.float32)
    return background.astype(np.float32)


def thicken_mask(mask: np.ndarray) -> np.ndarray:
    thick = mask.copy()
    thick |= np.roll(mask, 1, axis=1)
    thick |= np.roll(mask, -1, axis=1)
    thick[1:] |= mask[:-1]
    thick[:-1] |= mask[1:]
    return thick


def overlay_mask(background: np.ndarray, occupancy: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    image = np.asarray(background, dtype=np.float32).copy()
    thick = thicken_mask(occupancy)
    image[thick] = 0.55 * image[thick] + 0.45 * np.asarray(color, dtype=np.float32)
    image[occupancy] = np.asarray(color, dtype=np.float32)
    return np.clip(image, 0.0, 1.0)


def score_occupancy(score_map: np.ndarray, occupancy: np.ndarray) -> dict[str, float]:
    if not np.any(occupancy):
        return {
            "mean_score": 0.0,
            "p90_score": 0.0,
            "combined_score": 0.0,
            "coverage": 0.0,
            "edge_pixels": 0.0,
        }
    scores = score_map[occupancy]
    mean_score = float(np.mean(scores))
    p90_score = float(np.percentile(scores, 90.0))
    coverage = float(np.mean(occupancy))
    combined_score = mean_score + 0.2 * p90_score
    return {
        "mean_score": mean_score,
        "p90_score": p90_score,
        "combined_score": combined_score,
        "coverage": coverage,
        "edge_pixels": float(np.sum(occupancy)),
    }


def project_candidate(
    canonical_polylines: np.ndarray,
    candidate: AxisAlignment,
    stage: SearchStage,
 ) -> np.ndarray:
    transformed_polylines = apply_axis_alignment(canonical_polylines, candidate)
    return project_polylines_to_occupancy(
        transformed_polylines,
        resolution=stage.resolution,
        sample_factor=stage.sample_factor,
    )


def load_sample_artifacts(
    uid: str,
    sample_dir: Path,
    dataset: ObjaverseEdgeDataset,
    canonicalizer: MeshCanonicalizer,
    stages: list[SearchStage],
) -> SampleArtifacts:
    sample_path = sample_dir / f"{uid}.npz"
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found for {uid}: {sample_path}")

    sample = load_sample_modalities_local(sample_path)
    score_map = build_target_score_map(sample)
    score_maps = {stage.name: downsample_mean(score_map, stage.resolution) for stage in stages}
    background_image = build_depth_background(np.asarray(sample["model_depth"][0], dtype=np.float32))

    raw_polylines = dataset.load_edge_polylines(uid).astype(np.float32)
    canonical_polylines = canonicalizer.canonicalize_polylines(raw_polylines)

    return SampleArtifacts(
        uid=uid,
        sample_resolution=int(sample["resolution"]),
        score_maps=score_maps,
        background_image=background_image,
        full_score_map=score_map,
        canonical_polylines=canonical_polylines,
    )


def evaluate_stage(
    samples: list[SampleArtifacts],
    candidates: list[AxisAlignment],
    stage: SearchStage,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    total = len(samples) * len(candidates)
    completed = 0
    for candidate_index, candidate in enumerate(candidates, start=1):
        print(f"[{stage.name}] candidate {candidate_index:02d}/{len(candidates):02d}: {candidate_label(candidate)}")
        for sample in samples:
            occupancy = project_candidate(sample.canonical_polylines, candidate, stage)
            metrics = score_occupancy(sample.score_maps[stage.name], occupancy)
            completed += 1
            print(
                f"  [{completed:03d}/{total:03d}] {sample.uid} score={metrics['combined_score']:.6f} "
                f"mean={metrics['mean_score']:.6f} p90={metrics['p90_score']:.6f} coverage={metrics['coverage']:.6f}"
            )
            records.append(
                {
                    "stage": stage.name,
                    "uid": sample.uid,
                    "candidate_key": candidate_key(candidate),
                    "axis_order": list(candidate.axis_order),
                    "axis_signs": list(candidate.axis_signs),
                    "is_current": is_current_candidate(candidate),
                    **metrics,
                }
            )
    return records


def aggregate_stage(records: list[dict[str, object]]) -> list[dict[str, object]]:
    if not records:
        return []
    ranks_by_sample: dict[str, dict[str, int]] = {}
    sample_baselines: dict[str, float] = {}
    for uid in sorted({str(record["uid"]) for record in records}):
        sample_records = [record for record in records if record["uid"] == uid]
        sample_records.sort(key=lambda record: float(record["combined_score"]), reverse=True)
        ranks_by_sample[uid] = {
            str(record["candidate_key"]): rank for rank, record in enumerate(sample_records, start=1)
        }
        baseline_record = next(record for record in sample_records if bool(record["is_current"]))
        sample_baselines[uid] = float(baseline_record["combined_score"])

    aggregated: list[dict[str, object]] = []
    candidate_keys = sorted({str(record["candidate_key"]) for record in records})
    for key in candidate_keys:
        candidate_records = [record for record in records if record["candidate_key"] == key]
        first = candidate_records[0]
        ranks = [ranks_by_sample[str(record["uid"])][key] for record in candidate_records]
        deltas = [float(record["combined_score"]) - sample_baselines[str(record["uid"])] for record in candidate_records]
        aggregated.append(
            {
                "candidate_key": key,
                "axis_order": first["axis_order"],
                "axis_signs": first["axis_signs"],
                "is_current": first["is_current"],
                "mean_combined_score": float(np.mean([float(record["combined_score"]) for record in candidate_records])),
                "mean_score": float(np.mean([float(record["mean_score"]) for record in candidate_records])),
                "mean_p90_score": float(np.mean([float(record["p90_score"]) for record in candidate_records])),
                "mean_coverage": float(np.mean([float(record["coverage"]) for record in candidate_records])),
                "mean_rank": float(np.mean(ranks)),
                "win_count": int(sum(rank == 1 for rank in ranks)),
                "top3_count": int(sum(rank <= 3 for rank in ranks)),
                "mean_delta_vs_current": float(np.mean(deltas)),
            }
        )

    aggregated.sort(
        key=lambda record: (
            float(record["mean_rank"]),
            -float(record["mean_combined_score"]),
            -float(record["mean_delta_vs_current"]),
        )
    )
    return aggregated


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_from_aggregate_row(row: dict[str, object]) -> AxisAlignment:
    return AxisAlignment(axis_order=tuple(int(axis) for axis in row["axis_order"]), axis_signs=tuple(int(sign) for sign in row["axis_signs"]))


def save_visualizations(
    samples: list[SampleArtifacts],
    ranked_candidates: list[AxisAlignment],
    output_dir: Path,
    limit: int,
) -> None:
    visualization_stage = SearchStage(name="visualization", resolution=samples[0].sample_resolution, max_hits=3, sample_factor=2.0, depth_merge_tol=None)
    chosen_candidates = [CURRENT_EDGE_EXTRA_ALIGNMENT]
    for candidate in ranked_candidates:
        if candidate not in chosen_candidates:
            chosen_candidates.append(candidate)

    sample_subset = samples[: max(limit, 0)] if limit > 0 else samples
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)

    for sample in sample_subset:
        figure, axes = plt.subplots(len(chosen_candidates), 3, figsize=(18, 4 * len(chosen_candidates)), constrained_layout=True)
        if len(chosen_candidates) == 1:
            axes = np.expand_dims(axes, axis=0)
        for row_index, candidate in enumerate(chosen_candidates):
            occupancy = project_candidate(sample.canonical_polylines, candidate, visualization_stage)
            overlay = overlay_mask(sample.background_image, occupancy, color=(1.0, 0.15, 0.15))
            candidate_metrics = score_occupancy(sample.full_score_map, occupancy)

            axes[row_index, 0].imshow(sample.background_image)
            axes[row_index, 0].set_title("model depth hit1")
            axes[row_index, 1].imshow(sample.full_score_map, cmap="inferno")
            axes[row_index, 1].set_title("model-derived score map")
            axes[row_index, 2].imshow(overlay)
            axes[row_index, 2].set_title(
                f"{candidate_label(candidate)}\ncombined={candidate_metrics['combined_score']:.6f} mean={candidate_metrics['mean_score']:.6f}"
            )
            for axis in axes[row_index]:
                axis.axis("off")

        figure.suptitle(f"Edge transform comparison for {sample.uid}", fontsize=16)
        figure.savefig(visual_dir / f"{sample.uid}_comparison.png", dpi=150)
        plt.close(figure)


def print_top_candidates(title: str, rows: list[dict[str, object]], limit: int = 8) -> None:
    print(title)
    for index, row in enumerate(rows[:limit], start=1):
        print(
            f"  {index:02d}. {row['candidate_key']} order={tuple(row['axis_order'])} signs={tuple(row['axis_signs'])} "
            f"mean_rank={row['mean_rank']:.3f} mean_combined={row['mean_combined_score']:.6f} "
            f"delta_vs_current={row['mean_delta_vs_current']:.6f} wins={row['win_count']}"
        )


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coarse_stage = SearchStage(
        name="coarse",
        resolution=int(args.coarse_resolution),
        max_hits=int(args.coarse_max_hits),
        sample_factor=float(args.coarse_sample_factor),
        depth_merge_tol=2.0 / max(int(args.coarse_resolution), 1),
    )
    fine_stage = SearchStage(
        name="fine",
        resolution=int(args.fine_resolution),
        max_hits=int(args.fine_max_hits),
        sample_factor=float(args.fine_sample_factor),
        depth_merge_tol=2.0 / max(int(args.fine_resolution), 1),
    )

    uids = resolve_uids(sample_dir, args.uids, int(args.limit), args.reference_uid)
    if not uids:
        raise ValueError(f"No UIDs found under {sample_dir}")

    dataset = ObjaverseEdgeDataset(args.dataset_root)
    canonicalizer = MeshCanonicalizer(DEFAULT_EDGE3D_ALIGNMENT)
    samples = [load_sample_artifacts(uid, sample_dir, dataset, canonicalizer, [coarse_stage, fine_stage]) for uid in uids]
    candidates = enumerate_candidates()

    print(f"Loaded {len(samples)} samples for search")
    print("Sample UIDs:")
    for uid in uids:
        print(f"  - {uid}")

    coarse_records = evaluate_stage(samples, candidates, coarse_stage)
    coarse_aggregate = aggregate_stage(coarse_records)
    print_top_candidates("Coarse ranking:", coarse_aggregate)

    refine_count = max(int(args.refine_top_k), 1)
    fine_candidates = [candidate_from_aggregate_row(row) for row in coarse_aggregate[:refine_count]]
    if not any(is_current_candidate(candidate) for candidate in fine_candidates):
        fine_candidates.append(CURRENT_EDGE_EXTRA_ALIGNMENT)

    fine_records = evaluate_stage(samples, fine_candidates, fine_stage)
    fine_aggregate = aggregate_stage(fine_records)
    print_top_candidates("Fine ranking:", fine_aggregate)

    winning_row = fine_aggregate[0]
    winning_alignment = candidate_from_aggregate_row(winning_row)

    write_csv(output_dir / "coarse_scores.csv", coarse_records)
    write_csv(output_dir / "coarse_aggregate.csv", coarse_aggregate)
    write_csv(output_dir / "fine_scores.csv", fine_records)
    write_csv(output_dir / "fine_aggregate.csv", fine_aggregate)

    visualize_top_k = max(int(args.visualize_top_k), 1)
    visualization_candidates = [candidate_from_aggregate_row(row) for row in fine_aggregate[:visualize_top_k]]
    save_visualizations(samples, visualization_candidates, output_dir, limit=int(args.visualize_limit))

    summary = {
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "sample_dir": str(sample_dir),
        "uids": uids,
        "current_edge_extra_alignment": asdict(CURRENT_EDGE_EXTRA_ALIGNMENT),
        "winning_edge_extra_alignment": asdict(winning_alignment),
        "coarse_stage": asdict(coarse_stage),
        "fine_stage": asdict(fine_stage),
        "coarse_top_candidates": coarse_aggregate[: min(10, len(coarse_aggregate))],
        "fine_top_candidates": fine_aggregate[: min(10, len(fine_aggregate))],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Winning candidate:")
    print(f"  {candidate_label(winning_alignment)}")
    print(f"  summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()