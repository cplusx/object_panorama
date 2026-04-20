import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

try:
    import warp as wp
except ImportError:
    wp = None


@dataclass
class InverseSphericalRepresentation:
    colors: torch.Tensor
    radii: torch.Tensor
    inverse_radii: torch.Tensor
    valid_mask: torch.Tensor
    triangle_ids: torch.Tensor
    directions: torch.Tensor
    normals: torch.Tensor | None = None


def equirectangular_direction_map(height: int, device: str | torch.device = "cpu") -> torch.Tensor:
    width = 2 * height
    phi = torch.linspace(0, math.pi, height, device=device).view(height, 1)
    theta = torch.linspace(-math.pi, math.pi, width, device=device).view(1, width)
    x = torch.sin(phi) * torch.sin(theta)
    y = torch.cos(phi).repeat(1, width)
    z = torch.sin(phi) * torch.cos(theta)
    directions = torch.stack([x, y, z], dim=-1)
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def build_inward_equirectangular_rays(height: int, outer_radius: float = 5.0):
    directions = equirectangular_direction_map(height).cpu().numpy().astype(np.float32)
    origins = (outer_radius * directions).reshape(-1, 3)
    ray_directions = (-directions).reshape(-1, 3)
    return origins, ray_directions, directions


def _project_points_to_equirectangular_pixels(points: np.ndarray, height: int):
    width = 2 * height
    radii = np.linalg.norm(points, axis=1)
    keep = radii > 1e-8
    points = points[keep]
    radii = radii[keep]
    if len(points) == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    directions = points / radii[:, None]
    phi = np.arccos(np.clip(directions[:, 1], -1.0, 1.0))
    theta = np.arctan2(directions[:, 0], directions[:, 2])
    rows = np.clip((phi / math.pi * height).astype(np.int32), 0, height - 1)
    cols = np.clip(((theta + math.pi) / (2.0 * math.pi) * width).astype(np.int32), 0, width - 1)
    return rows, cols, radii.astype(np.float32)


def _wrapped_column_delta(start_col: int, end_col: int, width: int) -> int:
    delta = int(end_col) - int(start_col)
    half_width = width // 2
    if delta > half_width:
        delta -= width
    elif delta < -half_width:
        delta += width
    return delta


def _rasterize_wrapped_pixel_line(
    start_row: int,
    start_col: int,
    start_depth: float,
    end_row: int,
    end_col: int,
    end_depth: float,
    height: int,
    width: int,
):
    delta_row = int(end_row) - int(start_row)
    delta_col = _wrapped_column_delta(int(start_col), int(end_col), width)
    steps = max(abs(delta_row), abs(delta_col))
    if steps == 0:
        yield int(np.clip(start_row, 0, height - 1)), int(start_col) % width, float(start_depth)
        return

    last_key = None
    for step in range(steps + 1):
        t = step / steps
        row = int(np.clip(round(start_row + delta_row * t), 0, height - 1))
        col = int(round(start_col + delta_col * t)) % width
        key = (row, col)
        if key == last_key:
            continue
        depth = float((1.0 - t) * start_depth + t * end_depth)
        yield row, col, depth
        last_key = key


def polylines_to_inverse_spherical_representation(
    polylines: np.ndarray,
    resolution: int = 128,
    max_hits: int = 5,
    edge_color: tuple[int, int, int] = (0, 190, 255),
    sample_factor: float = 2.0,
    depth_merge_tol: float | None = None,
    device: str | torch.device = "cpu",
) -> InverseSphericalRepresentation:
    height = int(resolution)
    width = 2 * height
    max_hits = int(max_hits)
    sample_factor = float(sample_factor)
    depth_merge_tol = float(depth_merge_tol) if depth_merge_tol is not None else 2.0 / height

    color = np.asarray(edge_color, dtype=np.float32)
    if color.max() > 1.0:
        color = color / 255.0

    pixel_depths: dict[tuple[int, int], list[float]] = {}
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
            rows, cols, radii = _project_points_to_equirectangular_pixels(segment_points, height)
            if len(rows) == 0:
                continue
            if len(rows) == 1:
                pixel_depths.setdefault((int(rows[0]), int(cols[0])), []).append(float(radii[0]))
                continue

            for row0, col0, radius0, row1, col1, radius1 in zip(
                rows[:-1],
                cols[:-1],
                radii[:-1],
                rows[1:],
                cols[1:],
                radii[1:],
            ):
                for raster_row, raster_col, raster_depth in _rasterize_wrapped_pixel_line(
                    int(row0),
                    int(col0),
                    float(radius0),
                    int(row1),
                    int(col1),
                    float(radius1),
                    height=height,
                    width=width,
                ):
                    pixel_depths.setdefault((raster_row, raster_col), []).append(raster_depth)

    radii = torch.full((max_hits, height, width), float("nan"), dtype=torch.float32)
    inverse_radii = torch.zeros((max_hits, height, width), dtype=torch.float32)
    valid_mask = torch.zeros((max_hits, height, width), dtype=torch.bool)
    triangle_ids = torch.full((max_hits, height, width), -1, dtype=torch.int64)
    colors = torch.zeros((max_hits, 3, height, width), dtype=torch.float32)

    for (row, col), depths in pixel_depths.items():
        merged_depths: list[float] = []
        for depth in sorted(depths, reverse=True):
            if not merged_depths or abs(depth - merged_depths[-1]) > depth_merge_tol:
                merged_depths.append(depth)
        for layer, depth in enumerate(merged_depths[:max_hits]):
            radii[layer, row, col] = float(depth)
            inverse_radii[layer, row, col] = 1.0 / max(float(depth), 1e-8)
            valid_mask[layer, row, col] = True
            colors[layer, :, row, col] = torch.from_numpy(color)

    directions = equirectangular_direction_map(height, device=device).cpu()
    return InverseSphericalRepresentation(
        colors=colors,
        radii=radii,
        inverse_radii=inverse_radii,
        valid_mask=valid_mask,
        triangle_ids=triangle_ids,
        directions=directions,
        normals=None,
    )


def _load_trimesh(mesh_or_path: str | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(mesh_or_path, trimesh.Trimesh):
        return mesh_or_path

    mesh = trimesh.load(mesh_or_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if len(meshes) != 1:
            raise ValueError("Expected a single textured mesh when loading a scene.")
        mesh = meshes[0]
    return mesh


def trace_mesh_multi_hit(
    mesh: trimesh.Trimesh,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    max_hits: int = 4,
    batch_size: int = 512,
    stop_at_origin: bool = True,
    depth_epsilon: float = 1e-6,
):
    num_rays = ray_origins.shape[0]
    locations = np.full((max_hits, num_rays, 3), np.nan, dtype=np.float32)
    radii = np.full((max_hits, num_rays), np.nan, dtype=np.float32)
    triangle_ids = np.full((max_hits, num_rays), -1, dtype=np.int32)
    hit_counts = np.zeros(num_rays, dtype=np.int32)

    for start in range(0, num_rays, batch_size):
        end = min(start + batch_size, num_rays)
        batch_locs, batch_ray_ids, batch_tri_ids = mesh.ray.intersects_location(
            ray_origins=ray_origins[start:end],
            ray_directions=ray_directions[start:end],
            multiple_hits=True,
        )
        if len(batch_locs) == 0:
            continue

        batch_ray_ids = batch_ray_ids + start
        batch_depth = np.linalg.norm(batch_locs - ray_origins[batch_ray_ids], axis=1)
        batch_radii = np.linalg.norm(batch_locs, axis=1)

        if stop_at_origin:
            batch_max_depth = np.linalg.norm(ray_origins[batch_ray_ids], axis=1) + depth_epsilon
            keep = batch_depth <= batch_max_depth
            if not np.any(keep):
                continue
            batch_locs = batch_locs[keep]
            batch_ray_ids = batch_ray_ids[keep]
            batch_tri_ids = batch_tri_ids[keep]
            batch_radii = batch_radii[keep]
            batch_depth = batch_depth[keep]

        order = np.lexsort((batch_depth, batch_ray_ids))
        batch_locs = batch_locs[order]
        batch_ray_ids = batch_ray_ids[order]
        batch_tri_ids = batch_tri_ids[order]
        batch_radii = batch_radii[order]

        for loc, radius, ray_id, tri_id in zip(batch_locs, batch_radii, batch_ray_ids, batch_tri_ids):
            slot = hit_counts[ray_id]
            if slot >= max_hits:
                continue
            locations[slot, ray_id] = loc
            radii[slot, ray_id] = radius
            triangle_ids[slot, ray_id] = tri_id
            hit_counts[ray_id] += 1

    return locations, radii, triangle_ids


if wp is not None:

    @wp.kernel
    def _trace_mesh_multi_hit_warp_kernel(
        mesh_id: wp.uint64,
        ray_origins: wp.array(dtype=wp.vec3),
        ray_directions: wp.array(dtype=wp.vec3),
        ray_max_t: wp.array(dtype=wp.float32),
        max_hits: int,
        step_epsilon: float,
        hit_positions: wp.array(dtype=wp.vec3),
        hit_radii: wp.array(dtype=wp.float32),
        hit_faces: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        ray_count = ray_origins.shape[0]

        current_start = ray_origins[tid]
        ray_direction = ray_directions[tid]
        remaining_total = ray_max_t[tid]
        travelled = float(0.0)

        for slot in range(max_hits):
            remaining = remaining_total - travelled
            if remaining <= 0.0:
                break

            query = wp.mesh_query_ray(mesh_id, current_start, ray_direction, remaining)
            if not query.result:
                break

            out_idx = slot * ray_count + tid
            hit_position = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            hit_positions[out_idx] = hit_position
            hit_radii[out_idx] = wp.length(hit_position)
            hit_faces[out_idx] = query.face

            step = query.t + step_epsilon
            travelled = travelled + step
            current_start = current_start + ray_direction * step


def _normalize_cuda_device(device: str | torch.device) -> str:
    device_str = str(device)
    if device_str == "cuda":
        return "cuda:0"
    return device_str


def _ensure_warp_gpu_exact_available(device: str | torch.device) -> str:
    if wp is None:
        raise ImportError("gpu_exact backend requires warp-lang to be installed")

    resolved_device = _normalize_cuda_device(device)
    if not resolved_device.startswith("cuda"):
        raise ValueError("gpu_exact backend requires a CUDA device")

    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError(
            "warp CUDA backend is unavailable; check the CUDA toolkit/driver compatibility for warp-lang"
        )
    return resolved_device


def _deduplicate_consecutive_hits(
    hit_locations: np.ndarray,
    radii: np.ndarray,
    triangle_ids: np.ndarray,
    location_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = triangle_ids >= 0
    if hit_locations.shape[0] <= 1 or not valid.any():
        return hit_locations, radii, triangle_ids

    keep = valid.copy()
    consecutive_deltas = np.linalg.norm(hit_locations[1:] - hit_locations[:-1], axis=2)
    same_face = valid[1:] & valid[:-1] & (triangle_ids[1:] == triangle_ids[:-1])
    same_location = valid[1:] & valid[:-1] & (consecutive_deltas <= location_epsilon)
    keep[1:] &= ~(same_face | same_location)

    if np.array_equal(keep, valid):
        return hit_locations, radii, triangle_ids

    compact_locations = np.full_like(hit_locations, np.nan)
    compact_radii = np.full_like(radii, np.nan)
    compact_triangle_ids = np.full_like(triangle_ids, -1)
    write_slots = np.cumsum(keep, axis=0, dtype=np.int32) - 1

    for slot in range(hit_locations.shape[0]):
        ray_mask = keep[slot]
        if not np.any(ray_mask):
            continue
        ray_ids = np.flatnonzero(ray_mask)
        compact_slots = write_slots[slot, ray_ids]
        compact_locations[compact_slots, ray_ids] = hit_locations[slot, ray_ids]
        compact_radii[compact_slots, ray_ids] = radii[slot, ray_ids]
        compact_triangle_ids[compact_slots, ray_ids] = triangle_ids[slot, ray_ids]

    return compact_locations, compact_radii, compact_triangle_ids


def trace_mesh_multi_hit_warp(
    mesh: trimesh.Trimesh,
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    max_hits: int = 4,
    stop_at_origin: bool = True,
    depth_epsilon: float = 1e-6,
    device: str | torch.device = "cuda:0",
):
    resolved_device = _ensure_warp_gpu_exact_available(device)

    ray_origins = np.asarray(ray_origins, dtype=np.float32)
    ray_directions = np.asarray(ray_directions, dtype=np.float32)
    num_rays = int(ray_origins.shape[0])

    ray_max_t = np.full((num_rays,), np.float32(1.0e10), dtype=np.float32)
    if stop_at_origin:
        ray_max_t = np.linalg.norm(ray_origins, axis=1).astype(np.float32) + np.float32(depth_epsilon)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
    mesh_wp = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=resolved_device),
        indices=wp.array(faces, dtype=wp.int32, device=resolved_device),
        bvh_constructor="lbvh",
    )

    ray_origins_wp = wp.array(ray_origins, dtype=wp.vec3, device=resolved_device)
    ray_directions_wp = wp.array(ray_directions, dtype=wp.vec3, device=resolved_device)
    ray_max_t_wp = wp.array(ray_max_t, dtype=wp.float32, device=resolved_device)

    hit_positions_wp = wp.zeros(max_hits * num_rays, dtype=wp.vec3, device=resolved_device)
    hit_radii_wp = wp.zeros(max_hits * num_rays, dtype=wp.float32, device=resolved_device)
    hit_faces_wp = wp.array(
        np.full((max_hits * num_rays,), -1, dtype=np.int32),
        dtype=wp.int32,
        device=resolved_device,
    )

    step_epsilon = float(max(depth_epsilon, 1e-6))
    wp.launch(
        kernel=_trace_mesh_multi_hit_warp_kernel,
        dim=num_rays,
        inputs=[
            mesh_wp.id,
            ray_origins_wp,
            ray_directions_wp,
            ray_max_t_wp,
            int(max_hits),
            step_epsilon,
            hit_positions_wp,
            hit_radii_wp,
            hit_faces_wp,
        ],
        device=resolved_device,
    )
    wp.synchronize_device(resolved_device)

    hit_locations = hit_positions_wp.numpy().reshape(max_hits, num_rays, 3).astype(np.float32)
    radii = hit_radii_wp.numpy().reshape(max_hits, num_rays).astype(np.float32)
    triangle_ids = hit_faces_wp.numpy().reshape(max_hits, num_rays).astype(np.int32)

    hit_locations, radii, triangle_ids = _deduplicate_consecutive_hits(
        hit_locations,
        radii,
        triangle_ids,
        location_epsilon=float(max(depth_epsilon * 4.0, 1e-6)),
    )

    invalid = triangle_ids < 0
    hit_locations[invalid] = np.nan
    radii[invalid] = np.nan
    return hit_locations, radii, triangle_ids


def _barycentric_coordinates(points: torch.Tensor, triangle_vertices: torch.Tensor) -> torch.Tensor:
    v0, v1, v2 = triangle_vertices.unbind(1)
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    d00 = (v0v1 * v0v1).sum(1)
    d01 = (v0v1 * v0v2).sum(1)
    d11 = (v0v2 * v0v2).sum(1)
    d20 = ((points - v0) * v0v1).sum(1)
    d21 = ((points - v0) * v0v2).sum(1)
    denom = (d00 * d11 - d01 * d01).clamp_min(1e-12)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return torch.stack([u, v, w], dim=1).clamp(0.0, 1.0)


def sample_hit_attributes(
    mesh: trimesh.Trimesh,
    hit_locations: np.ndarray,
    triangle_ids: np.ndarray,
    ray_directions: np.ndarray,
    device: str | torch.device = "cpu",
    shading: str = "headlight",
) -> tuple[torch.Tensor, torch.Tensor]:
    layers, num_rays = triangle_ids.shape
    colors = torch.zeros((layers, num_rays, 3), dtype=torch.float32)
    normal_maps = torch.zeros((layers, num_rays, 3), dtype=torch.float32)

    valid_mask = triangle_ids >= 0
    if not valid_mask.any():
        return colors, normal_maps

    flat_points = torch.from_numpy(hit_locations[valid_mask]).to(device=device, dtype=torch.float32)
    flat_tri_ids = torch.from_numpy(triangle_ids[valid_mask]).to(device=device, dtype=torch.long)

    ray_ids = np.broadcast_to(np.arange(num_rays, dtype=np.int32)[None, :], (layers, num_rays))[valid_mask]
    flat_ray_dirs = torch.from_numpy(ray_directions[ray_ids]).to(device=device, dtype=torch.float32)

    vertices = torch.from_numpy(np.asarray(mesh.vertices).copy()).to(device=device, dtype=torch.float32)
    faces = torch.from_numpy(np.asarray(mesh.faces).copy()).to(device=device, dtype=torch.long)
    tri_vertices = vertices[faces[flat_tri_ids]]
    bary = _barycentric_coordinates(flat_points, tri_vertices)

    vertex_normals = torch.from_numpy(np.asarray(mesh.vertex_normals).copy()).to(device=device, dtype=torch.float32)
    tri_normals = vertex_normals[faces[flat_tri_ids]]
    normals = (tri_normals * bary[..., None]).sum(1)
    normals = F.normalize(normals, dim=1)

    material = getattr(mesh.visual, "material", None)
    uv = getattr(mesh.visual, "uv", None)

    if (
        isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals)
        and material is not None
        and getattr(material, "image", None) is not None
        and uv is not None
    ):
        texture_image = mesh.visual.material.image.convert("RGB")
        texture_np = np.asarray(texture_image)[..., :3].copy()
        texture_map = (
            torch.from_numpy(texture_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
            / 255.0
        )
        uv = torch.from_numpy(np.asarray(mesh.visual.uv).copy()).to(device=device, dtype=torch.float32)
        tri_uv = uv[faces[flat_tri_ids]]
        hit_uv = (tri_uv * bary[..., None]).sum(1)
        grid = torch.stack([hit_uv[:, 0] * 2 - 1, (1 - hit_uv[:, 1]) * 2 - 1], dim=1)
        sampled = F.grid_sample(
            texture_map,
            grid.view(1, -1, 1, 2),
            mode="bilinear",
            align_corners=True,
        )
        albedo = sampled.view(3, -1).t()
    elif hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
        face_colors = torch.from_numpy(np.asarray(mesh.visual.face_colors).copy()[:, :3]).to(device=device, dtype=torch.float32) / 255.0
        albedo = face_colors[flat_tri_ids]
    elif hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vertex_colors = torch.from_numpy(np.asarray(mesh.visual.vertex_colors).copy()[:, :3]).to(device=device, dtype=torch.float32) / 255.0
        tri_colors = vertex_colors[faces[flat_tri_ids]]
        albedo = (tri_colors * bary[..., None]).sum(1)
    elif material is not None and getattr(material, "main_color", None) is not None:
        base_color = torch.tensor(np.asarray(material.main_color)[:3], device=device, dtype=torch.float32) / 255.0
        albedo = base_color.unsqueeze(0).expand(flat_points.shape[0], -1)
    else:
        albedo = torch.ones((flat_points.shape[0], 3), device=device, dtype=torch.float32)

    if shading == "none":
        shaded = albedo
    elif shading == "headlight":
        view_dir = F.normalize(-flat_ray_dirs, dim=1)
        cos_theta = (normals * view_dir).sum(1, keepdim=True).abs()
        shaded = 0.1 * albedo + cos_theta * albedo
    else:
        raise ValueError(f"Unknown shading mode: {shading}")

    flat_output = torch.zeros((layers * num_rays, 3), device=device, dtype=torch.float32)
    flat_normal_output = torch.zeros((layers * num_rays, 3), device=device, dtype=torch.float32)
    flat_indices = torch.from_numpy(np.flatnonzero(valid_mask.reshape(-1))).to(device=device, dtype=torch.long)
    flat_output[flat_indices] = shaded
    flat_normal_output[flat_indices] = normals
    return flat_output.view(layers, num_rays, 3).cpu(), flat_normal_output.view(layers, num_rays, 3).cpu()


def _mesh_to_inverse_spherical_representation_gpu_exact(
    mesh: trimesh.Trimesh,
    resolution: int,
    max_hits: int,
    outer_radius: float,
    batch_size: int,
    shading: str,
    device: str | torch.device,
    stop_at_origin: bool,
) -> InverseSphericalRepresentation:
    ray_origins, ray_directions, direction_map_np = build_inward_equirectangular_rays(
        resolution,
        outer_radius=outer_radius,
    )
    hit_locations, radii_np, triangle_ids_np = trace_mesh_multi_hit_warp(
        mesh,
        ray_origins,
        ray_directions,
        max_hits=max_hits,
        stop_at_origin=stop_at_origin,
        device=device,
    )
    colors, normals = sample_hit_attributes(
        mesh,
        hit_locations,
        triangle_ids_np,
        ray_directions,
        device=device,
        shading=shading,
    )

    width = 2 * resolution
    radii = torch.from_numpy(radii_np).view(max_hits, resolution, width)
    valid_mask = torch.isfinite(radii) & (radii > 1e-8)
    inverse_radii = torch.zeros_like(radii)
    inverse_radii[valid_mask] = 1.0 / radii[valid_mask]
    triangle_ids = torch.from_numpy(triangle_ids_np).view(max_hits, resolution, width)
    colors = colors.view(max_hits, resolution, width, 3).permute(0, 3, 1, 2)
    normals = normals.view(max_hits, resolution, width, 3).permute(0, 3, 1, 2)
    directions = torch.from_numpy(direction_map_np)

    return InverseSphericalRepresentation(
        colors=colors,
        radii=radii,
        inverse_radii=inverse_radii,
        valid_mask=valid_mask,
        triangle_ids=triangle_ids,
        directions=directions,
        normals=normals,
    )


def sample_hit_colors(
    mesh: trimesh.Trimesh,
    hit_locations: np.ndarray,
    triangle_ids: np.ndarray,
    ray_directions: np.ndarray,
    device: str | torch.device = "cpu",
    shading: str = "headlight",
) -> torch.Tensor:
    colors, _ = sample_hit_attributes(
        mesh,
        hit_locations,
        triangle_ids,
        ray_directions,
        device=device,
        shading=shading,
    )
    return colors


def mesh_to_inverse_spherical_representation(
    mesh_or_path: str | trimesh.Trimesh,
    resolution: int = 256,
    max_hits: int = 4,
    outer_radius: float = 5.0,
    batch_size: int = 512,
    shading: str = "headlight",
    device: str | torch.device = "cpu",
    stop_at_origin: bool = True,
    backend: str = "auto",
) -> InverseSphericalRepresentation:
    mesh = _load_trimesh(mesh_or_path)
    resolved_backend = backend
    if backend == "auto":
        resolved_backend = "gpu_exact" if str(device).startswith("cuda") else "cpu_exact"
    if resolved_backend == "gpu_exact":
        return _mesh_to_inverse_spherical_representation_gpu_exact(
            mesh,
            resolution=resolution,
            max_hits=max_hits,
            outer_radius=outer_radius,
            batch_size=batch_size,
            shading=shading,
            device=device,
            stop_at_origin=stop_at_origin,
        )
    if resolved_backend != "cpu_exact":
        raise ValueError(f"Unknown mesh representation backend: {backend}")

    ray_origins, ray_directions, direction_map_np = build_inward_equirectangular_rays(resolution, outer_radius=outer_radius)
    hit_locations, radii_np, triangle_ids_np = trace_mesh_multi_hit(
        mesh,
        ray_origins,
        ray_directions,
        max_hits=max_hits,
        batch_size=batch_size,
        stop_at_origin=stop_at_origin,
    )
    colors, normals = sample_hit_attributes(
        mesh,
        hit_locations,
        triangle_ids_np,
        ray_directions,
        device=device,
        shading=shading,
    )

    width = 2 * resolution
    radii = torch.from_numpy(radii_np).view(max_hits, resolution, width)
    valid_mask = torch.isfinite(radii) & (radii > 1e-8)
    inverse_radii = torch.zeros_like(radii)
    inverse_radii[valid_mask] = 1.0 / radii[valid_mask]
    triangle_ids = torch.from_numpy(triangle_ids_np).view(max_hits, resolution, width)
    colors = colors.view(max_hits, resolution, width, 3).permute(0, 3, 1, 2)
    normals = normals.view(max_hits, resolution, width, 3).permute(0, 3, 1, 2)
    directions = torch.from_numpy(direction_map_np)

    return InverseSphericalRepresentation(
        colors=colors,
        radii=radii,
        inverse_radii=inverse_radii,
        valid_mask=valid_mask,
        triangle_ids=triangle_ids,
        directions=directions,
        normals=normals,
    )


def inverse_spherical_layer_to_points(rep: InverseSphericalRepresentation, layer: int = 0) -> torch.Tensor:
    radii = rep.radii[layer]
    valid = rep.valid_mask[layer]
    directions = rep.directions.to(radii)
    points = directions * radii[..., None]
    return points[valid]