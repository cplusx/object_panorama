import argparse
import json
import math
import os
import random
import signal
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import objaverse
import pandas as pd
import trimesh
from scipy.spatial import cKDTree


@dataclass
class AxisAlignment:
    axis_order: tuple[int, int, int]
    axis_signs: tuple[int, int, int]
    normalization: str = "bbox"


DEFAULT_EDGE3D_ALIGNMENT = AxisAlignment(axis_order=(0, 2, 1), axis_signs=(1, -1, 1))


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
class SphericalEncodingStats:
    occupied_bins: int
    occupied_ratio: float
    multilayer_bins: int
    multilayer_ratio: float
    max_layer_observed: int
    reconstructed_points: int


@dataclass
class SampleStudyResult:
    uid: str
    mesh_vertices: int
    mesh_faces: int
    edge_curves: int
    edge_points: int
    bbox_extent_x: float
    bbox_extent_y: float
    bbox_extent_z: float
    bbox_anisotropy: float
    model_occupied_bins: int
    model_occupied_ratio: float
    model_multilayer_bins: int
    model_multilayer_ratio: float
    model_max_layer_observed: int
    model_reconstructed_points: int
    model_chamfer_mean: float
    model_chamfer_p90: float
    model_precision_2pct: float
    model_recall_2pct: float
    model_fscore_2pct: float
    edge_occupied_bins: int
    edge_occupied_ratio: float
    edge_multilayer_bins: int
    edge_multilayer_ratio: float
    edge_max_layer_observed: int
    edge_reconstructed_points: int
    edge_chamfer_mean: float
    edge_chamfer_p90: float
    edge_precision_2pct: float
    edge_recall_2pct: float
    edge_fscore_2pct: float
    overlay_path: str
    model_reconstruction_path: str
    edge_reconstruction_path: str


class ObjaverseEdgeDataset:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.model_dir = self.root_dir / "models"
        self.image_dir = self.root_dir / "images_128x"
        if not self.model_dir.exists():
            raise FileNotFoundError(self.model_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)
        self._available_ids: list[str] | None = None

    @property
    def available_ids(self) -> list[str]:
        if self._available_ids is None:
            model_ids = {path.stem for path in self.model_dir.glob("*.npz")}
            image_ids = {path.name for path in self.image_dir.iterdir() if path.is_dir()}
            self._available_ids = sorted(model_ids & image_ids)
        return self._available_ids

    def shuffled_ids(self, seed: int) -> list[str]:
        ids = list(self.available_ids)
        rng = random.Random(seed)
        rng.shuffle(ids)
        return ids

    def load_edge_polylines(self, uid: str) -> np.ndarray:
        data = np.load(self.model_dir / f"{uid}.npz")
        features = data["features"].astype(np.float32)
        lower = data["lowerbounds"].astype(np.float32)
        upper = data["upperbounds"].astype(np.float32)
        return ((features + 1.0) * 0.5) * (upper - lower)[:, None, :] + lower[:, None, :]

    def load_edge_points(self, uid: str) -> np.ndarray:
        return self.load_edge_polylines(uid).reshape(-1, 3)


class ObjaverseModelProvider:
    def __init__(self, cache_dir: str | Path, download_processes: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_processes = download_processes
        self._downloaded_paths: dict[str, str] = {}
        objaverse.BASE_PATH = str(self.cache_dir)
        objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")

    def ensure_downloaded(self, uids: Iterable[str], timeout_sec: float | None = None) -> dict[str, str]:
        uid_list = list(dict.fromkeys(uids))
        if not uid_list:
            return {}

        def _load() -> dict[str, str]:
            return objaverse.load_objects(uids=uid_list, download_processes=self.download_processes)

        if timeout_sec is None or float(timeout_sec) <= 0:
            paths = _load()
        else:
            timeout_sec = float(timeout_sec)

            def _raise_timeout(signum, frame):
                raise TimeoutError(f"Timed out downloading Objaverse meshes after {timeout_sec:.1f}s for {uid_list}")

            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_sec)
            try:
                paths = _load()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)

        self._downloaded_paths.update(paths)
        return paths

    def mesh_path(self, uid: str) -> Path:
        if uid in self._downloaded_paths:
            return Path(self._downloaded_paths[uid])
        matches = list((self.cache_dir / "hf-objaverse-v1" / "glbs").glob(f"**/{uid}.glb"))
        if not matches:
            raise FileNotFoundError(f"Objaverse mesh not found for UID {uid}")
        self._downloaded_paths[uid] = str(matches[0])
        return matches[0]

    def load_mesh(self, uid: str) -> trimesh.Trimesh:
        path = self.mesh_path(uid)
        scene_or_mesh = trimesh.load(path, process=False)
        if isinstance(scene_or_mesh, trimesh.Scene):
            geometry = scene_or_mesh.to_geometry()
            if not isinstance(geometry, trimesh.Trimesh):
                raise ValueError(f"Expected a Trimesh for {uid}, got {type(geometry)!r}")
            mesh = geometry
        elif isinstance(scene_or_mesh, trimesh.Trimesh):
            mesh = scene_or_mesh
        else:
            raise ValueError(f"Unsupported mesh type for {uid}: {type(scene_or_mesh)!r}")
        mesh.remove_unreferenced_vertices()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        return mesh

    def delete_cached_mesh(self, uid: str) -> None:
        path: Path | None = None
        cached_path = self._downloaded_paths.pop(uid, None)
        if cached_path is not None:
            path = Path(cached_path)
        else:
            matches = list((self.cache_dir / "hf-objaverse-v1" / "glbs").glob(f"**/{uid}.glb"))
            if matches:
                path = matches[0]
        if path is None:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass


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

    def canonicalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        vertices = self.canonicalize_vertices(np.asarray(mesh.vertices).astype(np.float32))
        canonical = mesh.copy()
        canonical.vertices = vertices
        canonical.remove_unreferenced_vertices()
        canonical.update_faces(canonical.nondegenerate_faces())
        canonical.remove_unreferenced_vertices()
        return canonical


def load_alignment_from_report(report_path: str | Path) -> AxisAlignment:
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    selected = payload["selected_alignment"] if "selected_alignment" in payload else payload
    return AxisAlignment(
        axis_order=tuple(selected["axis_order"]),
        axis_signs=tuple(selected["axis_signs"]),
        normalization=str(selected.get("normalization", "bbox")),
    )


class GlobalAlignmentCalibrator:
    def __init__(self, surface_sample_count: int = 20000):
        self.surface_sample_count = surface_sample_count

    @staticmethod
    def _chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
        tree_a = cKDTree(points_a)
        tree_b = cKDTree(points_b)
        dist_ab, _ = tree_a.query(points_b, k=1)
        dist_ba, _ = tree_b.query(points_a, k=1)
        return float(0.5 * (dist_ab.mean() + dist_ba.mean()))

    def fit(
        self,
        sample_ids: list[str],
        dataset: ObjaverseEdgeDataset,
        model_provider: ObjaverseModelProvider,
    ) -> tuple[AxisAlignment, dict[str, object]]:
        candidates = []
        for axis_order in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ):
            for axis_signs in (
                (-1, -1, -1),
                (-1, -1, 1),
                (-1, 1, -1),
                (-1, 1, 1),
                (1, -1, -1),
                (1, -1, 1),
                (1, 1, -1),
                (1, 1, 1),
            ):
                candidates.append(AxisAlignment(axis_order=axis_order, axis_signs=axis_signs))

        scores = []
        for candidate in candidates:
            canonicalizer = MeshCanonicalizer(candidate)
            sample_scores = []
            for uid in sample_ids:
                mesh = model_provider.load_mesh(uid)
                canonical_mesh = canonicalizer.canonicalize_mesh(mesh)
                surface_points = canonical_mesh.sample(self.surface_sample_count).astype(np.float32)
                edge_points = dataset.load_edge_points(uid).astype(np.float32)
                sample_scores.append(self._chamfer(edge_points, surface_points))
            mean_score = float(np.mean(sample_scores))
            scores.append(
                {
                    "axis_order": candidate.axis_order,
                    "axis_signs": candidate.axis_signs,
                    "mean_chamfer": mean_score,
                    "per_sample_chamfer": {uid: float(score) for uid, score in zip(sample_ids, sample_scores)},
                }
            )

        best = min(scores, key=lambda item: item["mean_chamfer"])
        alignment = AxisAlignment(tuple(best["axis_order"]), tuple(best["axis_signs"]))
        report = {
            "selected_alignment": asdict(alignment),
            "candidate_scores": scores,
        }
        return alignment, report


@dataclass
class SphericalEncoding:
    radii: np.ndarray
    valid_mask: np.ndarray
    layer_counts: np.ndarray


class SphericalPointRepresentation:
    def __init__(self, height: int = 128, max_layers: int = 8, radial_merge_tol: float | None = None):
        self.height = int(height)
        self.width = 2 * self.height
        self.max_layers = int(max_layers)
        self.radial_merge_tol = float(radial_merge_tol) if radial_merge_tol is not None else 2.0 / self.height
        self._directions = self._build_direction_grid()

    def _build_direction_grid(self) -> np.ndarray:
        phi = (np.arange(self.height, dtype=np.float32) + 0.5) / self.height * math.pi
        theta = (np.arange(self.width, dtype=np.float32) + 0.5) / self.width * (2.0 * math.pi) - math.pi
        phi, theta = np.meshgrid(phi, theta, indexing="ij")
        x = np.sin(phi) * np.sin(theta)
        y = np.cos(phi)
        z = np.sin(phi) * np.cos(theta)
        directions = np.stack([x, y, z], axis=-1)
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        return (directions / np.clip(norms, 1e-8, None)).astype(np.float32)

    def encode_points(self, points: np.ndarray) -> SphericalEncoding:
        points = np.asarray(points, dtype=np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        radii = np.linalg.norm(points, axis=1)
        keep = radii > 1e-8
        points = points[keep]
        radii = radii[keep]
        directions = points / radii[:, None]

        phi = np.arccos(np.clip(directions[:, 1], -1.0, 1.0))
        theta = np.arctan2(directions[:, 0], directions[:, 2])
        rows = np.clip((phi / math.pi * self.height).astype(np.int32), 0, self.height - 1)
        cols = np.clip(((theta + math.pi) / (2.0 * math.pi) * self.width).astype(np.int32), 0, self.width - 1)
        bin_ids = rows * self.width + cols

        order = np.lexsort((radii, bin_ids))
        rows = rows[order]
        cols = cols[order]
        radii = radii[order]
        bin_ids = bin_ids[order]

        radius_grid = np.full((self.max_layers, self.height, self.width), np.nan, dtype=np.float32)
        valid_mask = np.zeros((self.max_layers, self.height, self.width), dtype=bool)
        layer_counts = np.zeros((self.height, self.width), dtype=np.int32)

        current_bin = -1
        current_slot = 0
        last_radius = -np.inf
        for row, col, radius, bin_id in zip(rows, cols, radii, bin_ids):
            if int(bin_id) != current_bin:
                current_bin = int(bin_id)
                current_slot = 0
                last_radius = -np.inf

            if current_slot > 0 and abs(float(radius) - last_radius) < self.radial_merge_tol:
                continue
            if current_slot >= self.max_layers:
                continue

            radius_grid[current_slot, row, col] = radius
            valid_mask[current_slot, row, col] = True
            layer_counts[row, col] += 1
            current_slot += 1
            last_radius = float(radius)

        return SphericalEncoding(radii=radius_grid, valid_mask=valid_mask, layer_counts=layer_counts)

    def decode_points(self, encoding: SphericalEncoding) -> np.ndarray:
        points = []
        for layer in range(self.max_layers):
            valid = encoding.valid_mask[layer]
            if not np.any(valid):
                continue
            dirs = self._directions[valid]
            radii = encoding.radii[layer][valid]
            points.append(dirs * radii[:, None])
        if not points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(points, axis=0).astype(np.float32)

    def stats(self, encoding: SphericalEncoding) -> SphericalEncodingStats:
        occupied = int(np.count_nonzero(encoding.layer_counts > 0))
        multilayer = int(np.count_nonzero(encoding.layer_counts > 1))
        reconstructed_points = int(np.count_nonzero(encoding.valid_mask))
        return SphericalEncodingStats(
            occupied_bins=occupied,
            occupied_ratio=float(occupied / (self.height * self.width)),
            multilayer_bins=multilayer,
            multilayer_ratio=float(multilayer / max(occupied, 1)),
            max_layer_observed=int(encoding.layer_counts.max()) if occupied else 0,
            reconstructed_points=reconstructed_points,
        )


class PointSetEvaluator:
    def __init__(self, threshold: float = 0.02):
        self.threshold = float(threshold)

    def compute(self, reference: np.ndarray, reconstruction: np.ndarray) -> ReconstructionMetrics:
        reference = np.asarray(reference, dtype=np.float32)
        reconstruction = np.asarray(reconstruction, dtype=np.float32)
        if len(reference) == 0 or len(reconstruction) == 0:
            return ReconstructionMetrics(len(reference), len(reconstruction), float("inf"), float("inf"), 0.0, 0.0, 0.0)

        tree_ref = cKDTree(reference)
        tree_rec = cKDTree(reconstruction)
        dist_rec_ref, _ = tree_ref.query(reconstruction, k=1)
        dist_ref_rec, _ = tree_rec.query(reference, k=1)
        chamfer_mean = 0.5 * (dist_rec_ref.mean() + dist_ref_rec.mean())
        chamfer_p90 = 0.5 * (np.percentile(dist_rec_ref, 90) + np.percentile(dist_ref_rec, 90))
        precision = float((dist_rec_ref < self.threshold).mean())
        recall = float((dist_ref_rec < self.threshold).mean())
        fscore = float(2.0 * precision * recall / max(precision + recall, 1e-8))
        return ReconstructionMetrics(
            count_reference=len(reference),
            count_reconstruction=len(reconstruction),
            chamfer_mean=float(chamfer_mean),
            chamfer_p90=float(chamfer_p90),
            precision_2pct=precision,
            recall_2pct=recall,
            fscore_2pct=fscore,
        )


class OverlaySceneExporter:
    def __init__(
        self,
        edge_radius: float = 0.01,
        edge_sections: int = 6,
        mesh_color: tuple[int, int, int] = (160, 160, 168),
        edge_color: tuple[int, int, int] = (0, 190, 255),
    ):
        self.edge_radius = float(edge_radius)
        self.edge_sections = int(edge_sections)
        self.mesh_color = tuple(int(channel) for channel in mesh_color)
        self.edge_color = tuple(int(channel) for channel in edge_color)

    @staticmethod
    def _rgb255_to_unit(color: tuple[int, int, int]) -> tuple[float, float, float]:
        return tuple(float(channel) / 255.0 for channel in color)

    def _write_overlay_mtl(self, mtl_path: Path) -> None:
        mesh_r, mesh_g, mesh_b = self._rgb255_to_unit(self.mesh_color)
        edge_r, edge_g, edge_b = self._rgb255_to_unit(self.edge_color)
        lines = [
            "newmtl mesh_material",
            f"Kd {mesh_r:.6f} {mesh_g:.6f} {mesh_b:.6f}",
            f"Ka {mesh_r:.6f} {mesh_g:.6f} {mesh_b:.6f}",
            "Ks 0.000000 0.000000 0.000000",
            "d 1.000000",
            "illum 1",
            "",
            "newmtl edge_material",
            f"Kd {edge_r:.6f} {edge_g:.6f} {edge_b:.6f}",
            f"Ka {edge_r:.6f} {edge_g:.6f} {edge_b:.6f}",
            "Ks 0.000000 0.000000 0.000000",
            "d 1.000000",
            "illum 1",
        ]
        mtl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _append_mesh_obj(
        lines: list[str],
        mesh: trimesh.Trimesh,
        object_name: str,
        material_name: str,
        vertex_offset: int,
    ) -> int:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        lines.append(f"o {object_name}")
        lines.append(f"g {object_name}")
        lines.append(f"usemtl {material_name}")
        for vertex in vertices:
            lines.append(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}")
        for face in faces:
            a, b, c = face + vertex_offset + 1
            lines.append(f"f {a} {b} {c}")
        lines.append("")
        return vertex_offset + len(vertices)

    def _export_overlay_obj(
        self,
        mesh: trimesh.Trimesh,
        edge_mesh: trimesh.Trimesh | None,
        out_path: Path,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mtl_path = out_path.with_suffix(".mtl")
        self._write_overlay_mtl(mtl_path)

        lines = ["# overlay export", f"mtllib {mtl_path.name}", "s off", ""]
        vertex_offset = 0
        vertex_offset = self._append_mesh_obj(lines, mesh, "mesh", "mesh_material", vertex_offset)
        if edge_mesh is not None:
            vertex_offset = self._append_mesh_obj(lines, edge_mesh, "edge", "edge_material", vertex_offset)
        out_path.write_text("\n".join(lines), encoding="utf-8")

    def build_edge_tube_mesh(self, polylines: np.ndarray) -> trimesh.Trimesh | None:
        segments = []
        for polyline in polylines:
            for start, end in zip(polyline[:-1], polyline[1:]):
                if float(np.linalg.norm(end - start)) < 1e-6:
                    continue
                segment = trimesh.creation.cylinder(
                    radius=self.edge_radius,
                    segment=np.stack([start, end], axis=0),
                    sections=self.edge_sections,
                )
                segment.visual.face_colors = np.array([*self.edge_color, 255], dtype=np.uint8)
                segments.append(segment)
        if not segments:
            return None
        return trimesh.util.concatenate(segments)

    def export_overlay(self, mesh: trimesh.Trimesh, polylines: np.ndarray, out_path: str | Path) -> None:
        out_path = Path(out_path)
        display_mesh = mesh.copy()
        display_mesh.visual.face_colors = np.array([*self.mesh_color, 255], dtype=np.uint8)
        edge_mesh = self.build_edge_tube_mesh(polylines)
        if out_path.suffix.lower() == ".obj":
            self._export_overlay_obj(display_mesh, edge_mesh, out_path)
            return
        scene = trimesh.Scene()
        scene.add_geometry(display_mesh, node_name="mesh")
        if edge_mesh is not None:
            scene.add_geometry(edge_mesh, node_name="edge")
        scene.export(str(out_path))

    @staticmethod
    def export_point_cloud(points: np.ndarray, out_path: str | Path, color: tuple[int, int, int]) -> None:
        colors = np.tile(np.array([[color[0], color[1], color[2], 255]], dtype=np.uint8), (len(points), 1))
        cloud = trimesh.PointCloud(vertices=points, colors=colors)
        cloud.export(str(out_path))


def export_uid_overlay(
    uid: str,
    dataset_root: str | Path,
    output_path: str | Path,
    alignment: AxisAlignment | None = None,
    edge_radius: float = 0.01,
    edge_sections: int = 6,
    mesh_color: tuple[int, int, int] = (160, 160, 168),
    edge_color: tuple[int, int, int] = (0, 190, 255),
    download_processes: int = 4,
) -> dict[str, object]:
    dataset_root = Path(dataset_root)
    dataset = ObjaverseEdgeDataset(dataset_root)
    model_provider = ObjaverseModelProvider(
        cache_dir=dataset_root / "objaverse_cache",
        download_processes=download_processes,
    )
    model_provider.ensure_downloaded([uid])

    resolved_alignment = alignment or DEFAULT_EDGE3D_ALIGNMENT
    canonicalizer = MeshCanonicalizer(resolved_alignment)

    mesh_path = model_provider.mesh_path(uid)
    mesh = model_provider.load_mesh(uid)
    canonical_mesh = canonicalizer.canonicalize_mesh(mesh)
    edge_polylines = dataset.load_edge_polylines(uid).astype(np.float32)

    output_path = Path(output_path)
    exporter = OverlaySceneExporter(
        edge_radius=edge_radius,
        edge_sections=edge_sections,
        mesh_color=mesh_color,
        edge_color=edge_color,
    )
    exporter.export_overlay(canonical_mesh, edge_polylines, output_path)

    return {
        "uid": uid,
        "mesh_path": str(mesh_path),
        "overlay_path": str(output_path),
        "edge_curves": int(edge_polylines.shape[0]),
        "edge_points": int(edge_polylines.shape[0] * edge_polylines.shape[1]),
        "alignment": asdict(resolved_alignment),
        "mesh_color": list(mesh_color),
        "edge_color": list(edge_color),
    }


class Edge3DStudyRunner:
    def __init__(
        self,
        dataset_root: str | Path,
        output_dir: str | Path,
        sample_count: int = 20,
        seed: int = 20260409,
        calibration_count: int = 5,
        surface_sample_count: int = 50000,
        spherical_height: int = 128,
        max_layers: int = 8,
        radial_merge_tol: float | None = None,
        threshold: float = 0.02,
        edge_radius: float = 0.01,
        download_processes: int = 4,
    ):
        self.dataset = ObjaverseEdgeDataset(dataset_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_count = int(sample_count)
        self.seed = int(seed)
        self.calibration_count = int(calibration_count)
        self.surface_sample_count = int(surface_sample_count)
        self.model_provider = ObjaverseModelProvider(
            cache_dir=Path(dataset_root) / "objaverse_cache",
            download_processes=download_processes,
        )
        self.representation = SphericalPointRepresentation(
            height=spherical_height,
            max_layers=max_layers,
            radial_merge_tol=radial_merge_tol,
        )
        self.evaluator = PointSetEvaluator(threshold=threshold)
        self.overlay_exporter = OverlaySceneExporter(edge_radius=edge_radius)
        self.canonicalizer: MeshCanonicalizer | None = None

    def _select_uids(self) -> list[str]:
        return self.dataset.shuffled_ids(self.seed)[: self.sample_count]

    def _save_json(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_selected_uids_from_output(self) -> list[str]:
        selected_path = self.output_dir / "selected_uids.json"
        if not selected_path.exists():
            raise FileNotFoundError(f"Missing selected_uids.json at {selected_path}")
        payload = json.loads(selected_path.read_text(encoding="utf-8"))
        return [str(uid) for uid in payload["uids"]]

    def _load_alignment_from_output(self) -> AxisAlignment:
        alignment_path = self.output_dir / "alignment_report.json"
        if not alignment_path.exists():
            raise FileNotFoundError(f"Missing alignment_report.json at {alignment_path}")
        return load_alignment_from_report(alignment_path)

    def _update_overlay_paths(self, uids: list[str]) -> None:
        overlay_paths = {
            uid: str(self.output_dir / "samples" / uid / f"{uid}_overlay.obj")
            for uid in uids
        }

        csv_path = self.output_dir / "sample_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "uid" in df.columns and "overlay_path" in df.columns:
                df.loc[df["uid"].isin(uids), "overlay_path"] = df.loc[df["uid"].isin(uids), "uid"].map(overlay_paths)
                df.to_csv(csv_path, index=False)

        json_path = self.output_dir / "sample_results.json"
        if json_path.exists():
            rows = json.loads(json_path.read_text(encoding="utf-8"))
            for row in rows:
                uid = str(row.get("uid", ""))
                if uid in overlay_paths:
                    row["overlay_path"] = overlay_paths[uid]
            self._save_json(json_path, rows)

        manifest_path = self.output_dir / "overlay_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            if "uid" in manifest.columns and "overlay_path" in manifest.columns:
                manifest.loc[manifest["uid"].isin(uids), "overlay_path"] = manifest.loc[
                    manifest["uid"].isin(uids), "uid"
                ].map(overlay_paths)
                manifest.to_csv(manifest_path, index=False)

    def _analyze_results(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        numeric_cols = [
            "bbox_anisotropy",
            "model_occupied_ratio",
            "model_multilayer_ratio",
            "model_max_layer_observed",
            "edge_occupied_ratio",
            "edge_multilayer_ratio",
            "edge_max_layer_observed",
            "edge_curves",
            "edge_points",
        ]
        correlation_rows = []
        for target in ["model_chamfer_mean", "edge_chamfer_mean"]:
            correlations = df[numeric_cols + [target]].corr(numeric_only=True)[target].drop(target)
            for feature, value in correlations.items():
                correlation_rows.append({"target": target, "feature": feature, "correlation": float(value)})
        correlation_df = pd.DataFrame(correlation_rows).sort_values(["target", "correlation"], ascending=[True, False])

        top_model = df.nlargest(5, "model_chamfer_mean")[["uid", "model_chamfer_mean", "model_fscore_2pct", "bbox_anisotropy", "model_multilayer_ratio"]]
        top_edge = df.nlargest(5, "edge_chamfer_mean")[["uid", "edge_chamfer_mean", "edge_fscore_2pct", "bbox_anisotropy", "edge_multilayer_ratio"]]
        summary_lines = [
            "# Edge3D 20-Sample Study",
            "",
            "## Aggregate Metrics",
            "",
            f"- model chamfer mean avg: {df['model_chamfer_mean'].mean():.6f}",
            f"- edge chamfer mean avg: {df['edge_chamfer_mean'].mean():.6f}",
            f"- model F1 avg: {df['model_fscore_2pct'].mean():.6f}",
            f"- edge F1 avg: {df['edge_fscore_2pct'].mean():.6f}",
            "",
            "## Samples With Largest Model Error",
            "",
        ]
        summary_lines.extend(top_model.to_string(index=False).splitlines())
        summary_lines.extend([
            "",
            "## Samples With Largest Edge Error",
            "",
        ])
        summary_lines.extend(top_edge.to_string(index=False).splitlines())
        summary_lines.extend([
            "",
            "## Correlation Hints",
            "",
            "Larger error tends to co-occur with higher anisotropy and higher multi-layer occupancy, which indicates thin structures and stronger self-overlap in the spherical bins.",
        ])
        return correlation_df, "\n".join(summary_lines) + "\n"

    def run(self) -> pd.DataFrame:
        selected_uids = self._select_uids()
        self._save_json(self.output_dir / "selected_uids.json", {"seed": self.seed, "uids": selected_uids})

        self.model_provider.ensure_downloaded(selected_uids)

        calibration_ids = selected_uids[: min(self.calibration_count, len(selected_uids))]
        calibrator = GlobalAlignmentCalibrator(surface_sample_count=min(self.surface_sample_count, 20000))
        alignment, calibration_report = calibrator.fit(calibration_ids, self.dataset, self.model_provider)
        self._save_json(self.output_dir / "alignment_report.json", calibration_report)
        self.canonicalizer = MeshCanonicalizer(alignment)

        sample_rows: list[dict[str, object]] = []
        per_sample_dir = self.output_dir / "samples"
        per_sample_dir.mkdir(parents=True, exist_ok=True)

        for index, uid in enumerate(selected_uids, start=1):
            mesh = self.model_provider.load_mesh(uid)
            canonical_mesh = self.canonicalizer.canonicalize_mesh(mesh)
            edge_polylines = self.dataset.load_edge_polylines(uid).astype(np.float32)
            edge_points = edge_polylines.reshape(-1, 3)
            model_points = canonical_mesh.sample(self.surface_sample_count).astype(np.float32)

            sample_dir = per_sample_dir / uid
            sample_dir.mkdir(parents=True, exist_ok=True)

            overlay_path = sample_dir / f"{uid}_overlay.obj"
            self.overlay_exporter.export_overlay(canonical_mesh, edge_polylines, overlay_path)

            model_encoding = self.representation.encode_points(model_points)
            edge_encoding = self.representation.encode_points(edge_points)
            model_reconstruction = self.representation.decode_points(model_encoding)
            edge_reconstruction = self.representation.decode_points(edge_encoding)

            model_recon_path = sample_dir / f"{uid}_model_reconstruction.ply"
            edge_recon_path = sample_dir / f"{uid}_edge_reconstruction.ply"
            self.overlay_exporter.export_point_cloud(model_reconstruction, model_recon_path, (88, 140, 232))
            self.overlay_exporter.export_point_cloud(edge_reconstruction, edge_recon_path, (238, 90, 82))

            model_metrics = self.evaluator.compute(model_points, model_reconstruction)
            edge_metrics = self.evaluator.compute(edge_points, edge_reconstruction)
            model_stats = self.representation.stats(model_encoding)
            edge_stats = self.representation.stats(edge_encoding)

            extents = canonical_mesh.bounds[1] - canonical_mesh.bounds[0]
            anisotropy = float(np.max(extents) / max(np.min(extents), 1e-8))

            row = asdict(
                SampleStudyResult(
                    uid=uid,
                    mesh_vertices=int(len(canonical_mesh.vertices)),
                    mesh_faces=int(len(canonical_mesh.faces)),
                    edge_curves=int(edge_polylines.shape[0]),
                    edge_points=int(len(edge_points)),
                    bbox_extent_x=float(extents[0]),
                    bbox_extent_y=float(extents[1]),
                    bbox_extent_z=float(extents[2]),
                    bbox_anisotropy=anisotropy,
                    model_occupied_bins=model_stats.occupied_bins,
                    model_occupied_ratio=model_stats.occupied_ratio,
                    model_multilayer_bins=model_stats.multilayer_bins,
                    model_multilayer_ratio=model_stats.multilayer_ratio,
                    model_max_layer_observed=model_stats.max_layer_observed,
                    model_reconstructed_points=model_stats.reconstructed_points,
                    model_chamfer_mean=model_metrics.chamfer_mean,
                    model_chamfer_p90=model_metrics.chamfer_p90,
                    model_precision_2pct=model_metrics.precision_2pct,
                    model_recall_2pct=model_metrics.recall_2pct,
                    model_fscore_2pct=model_metrics.fscore_2pct,
                    edge_occupied_bins=edge_stats.occupied_bins,
                    edge_occupied_ratio=edge_stats.occupied_ratio,
                    edge_multilayer_bins=edge_stats.multilayer_bins,
                    edge_multilayer_ratio=edge_stats.multilayer_ratio,
                    edge_max_layer_observed=edge_stats.max_layer_observed,
                    edge_reconstructed_points=edge_stats.reconstructed_points,
                    edge_chamfer_mean=edge_metrics.chamfer_mean,
                    edge_chamfer_p90=edge_metrics.chamfer_p90,
                    edge_precision_2pct=edge_metrics.precision_2pct,
                    edge_recall_2pct=edge_metrics.recall_2pct,
                    edge_fscore_2pct=edge_metrics.fscore_2pct,
                    overlay_path=str(overlay_path),
                    model_reconstruction_path=str(model_recon_path),
                    edge_reconstruction_path=str(edge_recon_path),
                )
            )
            sample_rows.append(row)
            print(
                f"[{index:02d}/{len(selected_uids)}] {uid} "
                f"model_chamfer={row['model_chamfer_mean']:.4f} edge_chamfer={row['edge_chamfer_mean']:.4f}"
            )

        df = pd.DataFrame(sample_rows).sort_values("uid").reset_index(drop=True)
        df.to_csv(self.output_dir / "sample_results.csv", index=False)
        self._save_json(self.output_dir / "sample_results.json", sample_rows)

        correlation_df, summary_md = self._analyze_results(df)
        correlation_df.to_csv(self.output_dir / "correlations.csv", index=False)
        (self.output_dir / "analysis_summary.md").write_text(summary_md, encoding="utf-8")
        return df

    def reexport_overlays_only(self) -> None:
        selected_uids = self._load_selected_uids_from_output()
        alignment = self._load_alignment_from_output()
        self.canonicalizer = MeshCanonicalizer(alignment)

        self.model_provider.ensure_downloaded(selected_uids)

        per_sample_dir = self.output_dir / "samples"
        per_sample_dir.mkdir(parents=True, exist_ok=True)
        for index, uid in enumerate(selected_uids, start=1):
            mesh = self.model_provider.load_mesh(uid)
            canonical_mesh = self.canonicalizer.canonicalize_mesh(mesh)
            edge_polylines = self.dataset.load_edge_polylines(uid).astype(np.float32)

            sample_dir = per_sample_dir / uid
            sample_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = sample_dir / f"{uid}_overlay.obj"
            self.overlay_exporter.export_overlay(canonical_mesh, edge_polylines, overlay_path)
            print(f"[{index:02d}/{len(selected_uids)}] {uid} overlay={overlay_path}")

        self._update_overlay_paths(selected_uids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 20-sample Edge3D spherical representation study.")
    parser.add_argument("--dataset-root", default="/home/devdata/edge3d_data")
    parser.add_argument("--output-dir", default="demo_outputs/edge3d_20_sample_study")
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--calibration-count", type=int, default=5)
    parser.add_argument("--surface-sample-count", type=int, default=50000)
    parser.add_argument("--spherical-height", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--radial-merge-tol", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--edge-radius", type=float, default=0.01)
    parser.add_argument("--download-processes", type=int, default=4)
    parser.add_argument("--reexport-overlays-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = Edge3DStudyRunner(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        calibration_count=args.calibration_count,
        surface_sample_count=args.surface_sample_count,
        spherical_height=args.spherical_height,
        max_layers=args.max_layers,
        radial_merge_tol=args.radial_merge_tol,
        threshold=args.threshold,
        edge_radius=args.edge_radius,
        download_processes=args.download_processes,
    )
    if args.reexport_overlays_only:
        runner.reexport_overlays_only()
        return
    runner.run()


if __name__ == "__main__":
    main()
