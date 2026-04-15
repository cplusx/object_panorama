import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesUV,
    look_at_rotation,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree

from pano_utils import (
    DistShader,
    MultiHitHardPhongShader,
    NormalAngleShader,
    cubemap_to_equirectangular,
    flip_cubemap_to_fit,
    get_sampling_prob_from_normal_angle,
    pano_to_pointcloud,
)


@dataclass
class PointCloudStats:
    count: int
    nn_mean: float
    nn_std: float
    nn_cv: float
    nn_p90: float


@dataclass
class MeshStats:
    vertices: int
    faces: int
    watertight: bool
    components: int
    chamfer_mean: float
    chamfer_p90: float
    precision_2pct: float
    recall_2pct: float
    fscore_2pct: float
    out_path: str | None


def normalize_mesh(mesh: Meshes) -> Meshes:
    num_verts = mesh.verts_packed().shape[0]
    mesh = mesh.offset_verts(-mesh.verts_packed().mean(dim=0, keepdim=True).repeat(num_verts, 1))
    mesh = mesh.scale_verts(1.0 / max(mesh.verts_packed().max(dim=0)[0]).item())
    return mesh


def safe_shift_mesh(mesh: Meshes, threshold: float = 0.1, n_dirs: int = 2048) -> tuple[Meshes, float]:
    verts = mesh.verts_packed().detach().cpu().numpy().astype(np.float64)
    norms2 = np.einsum("ij,ij->i", verts, verts)
    need_shift = norms2 < threshold * threshold
    if not np.any(need_shift):
        return mesh, 0.0

    rng = np.random.default_rng(0)
    z = rng.uniform(-1.0, 1.0, n_dirs)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_dirs)
    r_xy = np.sqrt(1.0 - z**2)
    dirs = np.column_stack((r_xy * np.cos(theta), r_xy * np.sin(theta), z))

    best_t = np.inf
    best_u = None
    for direction in dirs:
        vu = verts @ direction
        disc = vu[need_shift] ** 2 + (threshold * threshold - norms2[need_shift])
        t_i = -vu[need_shift] + np.sqrt(disc)
        t_u = 0.0 if t_i.size == 0 else np.max(t_i)
        if t_u < best_t:
            best_t = t_u
            best_u = direction

    if best_u is None:
        return mesh, 0.0

    shift = torch.tensor(best_u * best_t, dtype=torch.float32, device=mesh.device).view(1, 3)
    mesh = mesh.offset_verts(shift.repeat(mesh.verts_packed().shape[0], 1))
    return mesh, float(best_t)


def pytorch3d_to_trimesh(mesh: Meshes) -> trimesh.Trimesh:
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def inverse_points(points: torch.Tensor) -> torch.Tensor:
    return points / (points.norm(dim=1, keepdim=True) ** 2).clamp_min(1e-8)


def inverse_points_np(points: np.ndarray) -> np.ndarray:
    norm2 = np.sum(points * points, axis=1, keepdims=True)
    norm2 = np.clip(norm2, 1e-8, None)
    return points / norm2


def mesh_to_panorama_layers(mesh: Meshes, image_size: int, max_hits: int, device: torch.device):
    camera_dirs = torch.tensor(
        [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    up_vectors = torch.tensor(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        device=device,
        dtype=torch.float32,
    )
    rotations = look_at_rotation(camera_dirs, up=up_vectors)
    translations = torch.zeros_like(camera_dirs)

    verts = mesh.verts_packed()
    inverted_verts = verts / (verts.norm(dim=1, keepdim=True) ** 2).clamp_min(1e-8)

    faces = mesh.faces_packed()
    flipped_faces = faces[:, [0, 2, 1]]
    verts_uvs = mesh.textures.verts_uvs_padded()
    faces_uvs = mesh.textures.faces_uvs_padded()
    flipped_fuvs = faces_uvs[..., [0, 2, 1]]

    textures = TexturesUV(
        maps=mesh.textures._maps_padded,
        faces_uvs=flipped_fuvs.to(device),
        verts_uvs=verts_uvs.to(device),
    )
    inverted_mesh = Meshes(verts=[inverted_verts], faces=[flipped_faces], textures=textures)

    cameras = FoVPerspectiveCameras(device=device, R=rotations, T=translations, fov=90, znear=0.1, zfar=200)
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=max_hits)
    lights = PointLights(device=device, location=[[0, 0, 0.0]])
    blend_params = BlendParams(background_color=(-1, -1, -1))

    color_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=MultiHitHardPhongShader(device=device, cameras=cameras, lights=lights, max_hits=max_hits),
    )
    dist_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=DistShader(cameras=cameras, max_hits=max_hits),
    )
    normal_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=NormalAngleShader(cameras=cameras, max_hits=max_hits),
    )

    color_layers = color_renderer(inverted_mesh.extend(6), cameras=cameras, lights=lights, blend_params=blend_params)
    dist_layers = dist_renderer(inverted_mesh.extend(6), cameras=cameras)
    normal_layers = normal_renderer(inverted_mesh.extend(6), cameras=cameras)

    panos = []
    valid_masks = []
    pano_dists = []
    pano_normals = []
    cube_color_layers = []
    cube_dist_layers = []

    for color_images, dist_images, normal_images in zip(color_layers, dist_layers, normal_layers):
        cube_colors = flip_cubemap_to_fit(color_images[..., :3].permute(0, 3, 1, 2))
        pano = cubemap_to_equirectangular(cube_colors, H=image_size, device=device)
        valid_mask = pano.mean(dim=0) > 0
        panos.append(pano)
        valid_masks.append(valid_mask)
        cube_color_layers.append(cube_colors)

        cube_dist = flip_cubemap_to_fit(dist_images.permute(0, 3, 1, 2)).squeeze(1)
        pano_dist = cubemap_to_equirectangular(cube_dist[:, None], H=image_size, device=device)[0]
        pano_dists.append(pano_dist)
        cube_dist_layers.append(cube_dist)

        cube_normal = flip_cubemap_to_fit(normal_images.permute(0, 3, 1, 2)).squeeze(1)
        pano_normal = cubemap_to_equirectangular(cube_normal[:, None], H=image_size, device=device)[0]
        pano_normals.append(pano_normal)

    return {
        "panos": panos,
        "valid_masks": valid_masks,
        "pano_dists": pano_dists,
        "pano_normals": pano_normals,
        "cube_colors": cube_color_layers,
        "cube_dists": cube_dist_layers,
    }


def pano_layers_to_point_cloud(data: dict, device: torch.device, sampling_mode: str = "raw", seed: int = 0):
    rng = np.random.default_rng(seed)
    image_size = data["panos"][0].shape[1]
    phi = torch.linspace(0, math.pi, image_size, device=device).view(image_size, 1)
    latitude_weight = (0.25 + 0.75 * torch.sin(phi)).repeat(1, 2 * image_size)

    all_points = []
    all_colors = []

    for pano, pano_dist, valid_mask, pano_normal in zip(
        data["panos"], data["pano_dists"], data["valid_masks"], data["pano_normals"]
    ):
        this_valid = valid_mask.clone()
        if sampling_mode == "angle_area":
            angle_prob = get_sampling_prob_from_normal_angle(pano_normal.abs(), max_prob=1.0, min_prob=0.15)
            keep_prob = (angle_prob * latitude_weight).clamp(0.05, 1.0)
            random_map = torch.from_numpy(rng.random(keep_prob.shape)).to(device=device, dtype=keep_prob.dtype)
            this_valid = this_valid & (random_map < keep_prob)

        points, colors = pano_to_pointcloud(pano, pano_dist, this_valid)
        points = inverse_points(points)
        all_points.append(points)
        all_colors.append(colors)

    points = torch.cat(all_points, dim=0)
    colors = torch.cat(all_colors, dim=0)
    return points.detach().cpu().numpy(), colors.detach().cpu().numpy()


def layered_pano_mesh(data: dict) -> trimesh.Trimesh:
    pano_color = torch.stack(data["panos"], dim=0)
    pano_dist = torch.stack(data["pano_dists"], dim=0)
    valid_mask = torch.stack(data["valid_masks"], dim=0)

    layers, _, height, width = pano_color.shape
    device = pano_color.device
    pano_color = pano_color.permute(0, 2, 3, 1)

    phi = torch.linspace(0, math.pi, height, device=device).view(height, 1)
    theta = torch.linspace(-math.pi, math.pi, width, device=device).view(1, width)
    x = torch.sin(phi) * torch.sin(theta)
    y = torch.cos(phi).repeat(1, width)
    z = torch.sin(phi) * torch.cos(theta)
    dirs = torch.stack([x, y, z], dim=-1).unsqueeze(0).expand(layers, -1, -1, -1)
    points = dirs * pano_dist.unsqueeze(-1)

    index_map = -torch.ones((layers, height, width), dtype=torch.long, device=device)
    valid_idx = valid_mask.nonzero(as_tuple=False)
    index_map[valid_mask] = torch.arange(valid_idx.shape[0], device=device)

    verts = points[valid_mask]
    colors = pano_color[valid_mask]
    faces = []

    def tri_from_quad(i00, i01, i10, i11):
        mask_a = (i00 >= 0) & (i01 >= 0) & (i10 >= 0)
        mask_b = (i11 >= 0) & (i01 >= 0) & (i10 >= 0)
        tri_a = torch.stack([i00[mask_a], i01[mask_a], i10[mask_a]], dim=-1)
        tri_b = torch.stack([i11[mask_b], i01[mask_b], i10[mask_b]], dim=-1)
        return [tri_a, tri_b]

    y_idx = torch.arange(height - 1, device=device)
    x_idx = torch.arange(width, device=device)
    yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")

    for layer in range(layers):
        i00 = index_map[layer, yy, xx]
        i01 = index_map[layer, yy, (xx + 1) % width]
        i10 = index_map[layer, yy + 1, xx]
        i11 = index_map[layer, yy + 1, (xx + 1) % width]
        faces += tri_from_quad(i00, i01, i10, i11)

    layer_idx = torch.arange(layers - 1, device=device)
    yy_full, xx_full = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")

    for layer in layer_idx.tolist():
        i00 = index_map[layer, yy_full, xx_full]
        i10 = index_map[layer + 1, yy_full, xx_full]
        i01 = index_map[layer, yy_full, (xx_full + 1) % width]
        i11 = index_map[layer + 1, yy_full, (xx_full + 1) % width]
        faces += tri_from_quad(i00, i01, i10, i11)

    faces = torch.cat(faces, dim=0)
    verts = inverse_points(verts)

    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    colors_np = (colors.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, vertex_colors=colors_np, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def exact_ray_hit_cloud(reference: trimesh.Trimesh, resolution: int, radius: float = 5.0, batch_size: int = 256):
    phi = (np.arange(resolution) + 0.5) / resolution * math.pi
    theta = (np.arange(2 * resolution) + 0.5) / (2 * resolution) * 2 * math.pi
    phi, theta = np.meshgrid(phi, theta, indexing="ij")

    outward = np.stack(
        [
            np.sin(phi) * np.cos(theta),
            np.cos(phi),
            np.sin(phi) * np.sin(theta),
        ],
        axis=-1,
    )
    origins = (radius * outward).reshape(-1, 3)
    directions = (-outward).reshape(-1, 3)

    all_hits = []
    for start in range(0, len(origins), batch_size):
        end = min(start + batch_size, len(origins))
        locs, ray_ids, _ = reference.ray.intersects_location(
            ray_origins=origins[start:end],
            ray_directions=directions[start:end],
            multiple_hits=True,
        )
        if len(locs) == 0:
            continue

        depths = np.linalg.norm(locs - origins[start:end][ray_ids], axis=1)
        max_depth = np.linalg.norm(origins[start:end][ray_ids], axis=1) + 1e-6
        keep = depths <= max_depth
        if np.any(keep):
            all_hits.append(locs[keep])

    if not all_hits:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    points = np.concatenate(all_hits, axis=0).astype(np.float32)
    colors = np.zeros_like(points, dtype=np.float32)
    return points, colors


def inverted_mesh_points(
    reference: trimesh.Trimesh,
    resolution: int,
    device: torch.device,
    subdivide_levels: int = 0,
):
    tri = reference.copy()
    for _ in range(subdivide_levels):
        tri = tri.subdivide()

    vertices = torch.from_numpy(np.asarray(tri.vertices)).float().to(device)
    faces = torch.from_numpy(np.asarray(tri.faces)).long().to(device)
    vertices = vertices / (vertices.norm(dim=1, keepdim=True) ** 2).clamp_min(1e-8)
    faces = faces[:, [0, 2, 1]]

    camera_dirs = torch.tensor(
        [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    up_vectors = torch.tensor(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        device=device,
        dtype=torch.float32,
    )
    rotations = look_at_rotation(camera_dirs, up=up_vectors)
    translations = torch.zeros_like(camera_dirs)
    cameras = FoVPerspectiveCameras(device=device, R=rotations, T=translations, fov=90, znear=0.01, zfar=500)
    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    mesh = Meshes(verts=[vertices], faces=[faces])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=DistShader(cameras=cameras, max_hits=1),
    )

    dist_maps = renderer(mesh.extend(6), cameras=cameras)[0]
    dist_maps = flip_cubemap_to_fit(dist_maps.permute(0, 3, 1, 2)).squeeze(1)
    pano_dist = cubemap_to_equirectangular(dist_maps[:, None], H=resolution, device=device)[0]
    valid_mask = torch.isfinite(pano_dist) & (pano_dist > 1e-6)
    dummy_color = torch.zeros(3, resolution, 2 * resolution, device=device)
    points_inv, _ = pano_to_pointcloud(dummy_color, pano_dist, valid_mask)
    points = inverse_points(points_inv).detach().cpu().numpy()
    colors = np.zeros_like(points, dtype=np.float32)
    return points, colors


def point_cloud_stats(points: np.ndarray, max_points: int = 20000) -> PointCloudStats:
    if len(points) == 0:
        return PointCloudStats(0, float("nan"), float("nan"), float("nan"), float("nan"))

    total_count = len(points)
    if len(points) > max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), size=max_points, replace=False)]

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    nn = dists[:, 1]
    mean = float(nn.mean())
    std = float(nn.std())
    cv = float(std / max(mean, 1e-8))
    return PointCloudStats(total_count, mean, std, cv, float(np.percentile(nn, 90)))


def numpy_to_o3d_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    cloud = cloud.remove_duplicated_points()
    cloud = cloud.remove_non_finite_points()
    return cloud


def estimate_normals(cloud: o3d.geometry.PointCloud, radius: float, max_nn: int = 64):
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    if len(cloud.points) >= 16:
        try:
            cloud.orient_normals_consistent_tangent_plane(min(64, len(cloud.points) - 1))
        except RuntimeError:
            cloud.orient_normals_towards_camera_location(camera_location=np.zeros(3))
    cloud.normalize_normals()
    return cloud


def poisson_mesh_from_cloud(cloud: o3d.geometry.PointCloud, depth: int, crop: bool = True) -> trimesh.Trimesh | None:
    if len(cloud.points) < 32:
        return None
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=depth)
    if crop:
        mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    if len(mesh.triangles) == 0:
        return None
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )


def bpa_mesh_from_cloud(cloud: o3d.geometry.PointCloud) -> trimesh.Trimesh | None:
    if len(cloud.points) < 32:
        return None
    nn = np.asarray(cloud.compute_nearest_neighbor_distance())
    if len(nn) == 0:
        return None
    avg_d = float(nn.mean())
    radii = o3d.utility.DoubleVector([avg_d * factor for factor in (1.2, 1.8, 2.4)])
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, radii)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    if len(mesh.triangles) == 0:
        return None
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        process=False,
    )


def sample_surface_points(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    if mesh is None or len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    samples, _ = trimesh.sample.sample_surface(mesh, count)
    return samples.astype(np.float32)


def mesh_metrics(recon: trimesh.Trimesh | None, reference: trimesh.Trimesh, threshold: float, out_path: str | None) -> MeshStats:
    if recon is None or len(recon.faces) == 0:
        return MeshStats(0, 0, False, 0, float("inf"), float("inf"), 0.0, 0.0, 0.0, out_path)

    ref_points = sample_surface_points(reference, 20000)
    rec_points = sample_surface_points(recon, 20000)
    if len(ref_points) == 0 or len(rec_points) == 0:
        return MeshStats(0, 0, False, 0, float("inf"), float("inf"), 0.0, 0.0, 0.0, out_path)

    ref_tree = cKDTree(ref_points)
    rec_tree = cKDTree(rec_points)
    d_rec_to_ref, _ = ref_tree.query(rec_points, k=1)
    d_ref_to_rec, _ = rec_tree.query(ref_points, k=1)
    chamfer = 0.5 * (d_rec_to_ref.mean() + d_ref_to_rec.mean())
    chamfer_p90 = 0.5 * (np.percentile(d_rec_to_ref, 90) + np.percentile(d_ref_to_rec, 90))

    precision = float((d_rec_to_ref < threshold).mean())
    recall = float((d_ref_to_rec < threshold).mean())
    fscore = 2 * precision * recall / max(precision + recall, 1e-8)

    components = len(recon.split(only_watertight=False))
    return MeshStats(
        vertices=int(len(recon.vertices)),
        faces=int(len(recon.faces)),
        watertight=bool(recon.is_watertight),
        components=int(components),
        chamfer_mean=float(chamfer),
        chamfer_p90=float(chamfer_p90),
        precision_2pct=precision,
        recall_2pct=recall,
        fscore_2pct=float(fscore),
        out_path=out_path,
    )


def save_preview(pano: torch.Tensor, out_path: str):
    image = (pano.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(image).save(out_path)


def run_sample(sample_name: str, mesh_path: str, args, device: torch.device, output_root: str):
    sample_dir = os.path.join(output_root, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    mesh = load_objs_as_meshes([mesh_path], device=device)
    mesh = normalize_mesh(mesh)
    mesh, shift = safe_shift_mesh(mesh)
    reference_mesh = pytorch3d_to_trimesh(mesh)

    pano_data = mesh_to_panorama_layers(mesh, image_size=args.image_size, max_hits=args.max_hits, device=device)
    save_preview(pano_data["panos"][0], os.path.join(sample_dir, "layer0_pano.png"))

    raw_points, raw_colors = pano_layers_to_point_cloud(pano_data, device=device, sampling_mode="raw", seed=args.seed)
    angle_points, angle_colors = pano_layers_to_point_cloud(
        pano_data, device=device, sampling_mode="angle_area", seed=args.seed
    )

    bbox_diag = np.linalg.norm(raw_points.max(axis=0) - raw_points.min(axis=0))
    reference_bbox_diag = np.linalg.norm(reference_mesh.vertices.max(axis=0) - reference_mesh.vertices.min(axis=0))
    voxel_size = reference_bbox_diag / args.voxel_divisor
    normal_radius = voxel_size * 2.5
    threshold = reference_bbox_diag * 0.02

    raw_cloud = numpy_to_o3d_cloud(raw_points, raw_colors)
    voxel_cloud = raw_cloud.voxel_down_sample(voxel_size)
    angle_cloud = numpy_to_o3d_cloud(angle_points, angle_colors).voxel_down_sample(voxel_size)

    raw_cloud = estimate_normals(raw_cloud, radius=normal_radius)
    voxel_cloud = estimate_normals(voxel_cloud, radius=normal_radius)
    angle_cloud = estimate_normals(angle_cloud, radius=normal_radius)

    methods = {}

    layered_mesh = layered_pano_mesh(pano_data)
    layered_path = os.path.join(sample_dir, "layered_pano_mesh.ply")
    layered_mesh.export(layered_path)
    methods["layered_pano_mesh"] = mesh_metrics(layered_mesh, reference_mesh, threshold, layered_path)

    vggt_mesh = poisson_mesh_from_cloud(raw_cloud, depth=args.poisson_depth)
    vggt_path = os.path.join(sample_dir, "vggt_like_poisson.ply")
    if vggt_mesh is not None:
        vggt_mesh.export(vggt_path)
    methods["vggt_like_poisson"] = mesh_metrics(vggt_mesh, reference_mesh, threshold, vggt_path if vggt_mesh else None)

    voxel_poisson_mesh = poisson_mesh_from_cloud(voxel_cloud, depth=args.poisson_depth)
    voxel_poisson_path = os.path.join(sample_dir, "voxel_poisson.ply")
    if voxel_poisson_mesh is not None:
        voxel_poisson_mesh.export(voxel_poisson_path)
    methods["voxel_poisson"] = mesh_metrics(
        voxel_poisson_mesh, reference_mesh, threshold, voxel_poisson_path if voxel_poisson_mesh else None
    )

    angle_poisson_mesh = poisson_mesh_from_cloud(angle_cloud, depth=args.poisson_depth)
    angle_poisson_path = os.path.join(sample_dir, "angle_area_poisson.ply")
    if angle_poisson_mesh is not None:
        angle_poisson_mesh.export(angle_poisson_path)
    methods["angle_area_poisson"] = mesh_metrics(
        angle_poisson_mesh, reference_mesh, threshold, angle_poisson_path if angle_poisson_mesh else None
    )

    bpa_mesh = bpa_mesh_from_cloud(voxel_cloud)
    bpa_path = os.path.join(sample_dir, "voxel_bpa.ply")
    if bpa_mesh is not None:
        bpa_mesh.export(bpa_path)
    methods["voxel_bpa"] = mesh_metrics(bpa_mesh, reference_mesh, threshold, bpa_path if bpa_mesh else None)

    if args.include_exact_ray:
        exact_points, exact_colors = exact_ray_hit_cloud(
            reference_mesh,
            resolution=args.exact_ray_resolution,
        )
        exact_cloud = numpy_to_o3d_cloud(exact_points, exact_colors)
        exact_cloud = exact_cloud.voxel_down_sample(voxel_size)
        exact_cloud = estimate_normals(exact_cloud, radius=normal_radius)

        exact_poisson = poisson_mesh_from_cloud(exact_cloud, depth=max(7, args.poisson_depth - 1))
        exact_poisson_path = os.path.join(sample_dir, "exact_ray_poisson.ply")
        if exact_poisson is not None:
            exact_poisson.export(exact_poisson_path)
        methods["exact_ray_poisson"] = mesh_metrics(
            exact_poisson,
            reference_mesh,
            threshold,
            exact_poisson_path if exact_poisson else None,
        )

        exact_bpa = bpa_mesh_from_cloud(exact_cloud)
        exact_bpa_path = os.path.join(sample_dir, "exact_ray_bpa.ply")
        if exact_bpa is not None:
            exact_bpa.export(exact_bpa_path)
        methods["exact_ray_bpa"] = mesh_metrics(
            exact_bpa,
            reference_mesh,
            threshold,
            exact_bpa_path if exact_bpa else None,
        )

    if args.include_subdivided_inversion:
        subdiv_points, subdiv_colors = inverted_mesh_points(
            reference_mesh,
            resolution=args.exact_ray_resolution,
            device=device,
            subdivide_levels=args.subdivide_levels,
        )
        subdiv_cloud = numpy_to_o3d_cloud(subdiv_points, subdiv_colors)
        subdiv_cloud = subdiv_cloud.voxel_down_sample(voxel_size)
        subdiv_cloud = estimate_normals(subdiv_cloud, radius=normal_radius)

        subdiv_poisson = poisson_mesh_from_cloud(subdiv_cloud, depth=max(7, args.poisson_depth - 1))
        subdiv_poisson_path = os.path.join(sample_dir, "subdivided_inversion_poisson.ply")
        if subdiv_poisson is not None:
            subdiv_poisson.export(subdiv_poisson_path)
        methods["subdivided_inversion_poisson"] = mesh_metrics(
            subdiv_poisson,
            reference_mesh,
            threshold,
            subdiv_poisson_path if subdiv_poisson else None,
        )

        subdiv_bpa = bpa_mesh_from_cloud(subdiv_cloud)
        subdiv_bpa_path = os.path.join(sample_dir, "subdivided_inversion_bpa.ply")
        if subdiv_bpa is not None:
            subdiv_bpa.export(subdiv_bpa_path)
        methods["subdivided_inversion_bpa"] = mesh_metrics(
            subdiv_bpa,
            reference_mesh,
            threshold,
            subdiv_bpa_path if subdiv_bpa else None,
        )

    result = {
        "mesh_path": mesh_path,
        "shift": shift,
        "bbox_diag": float(bbox_diag),
        "reference_bbox_diag": float(reference_bbox_diag),
        "voxel_size": float(voxel_size),
        "reference_vertices": int(len(reference_mesh.vertices)),
        "reference_faces": int(len(reference_mesh.faces)),
        "preview_path": os.path.join(sample_dir, "layer0_pano.png"),
        "point_clouds": {
            "raw": asdict(point_cloud_stats(raw_points)),
            "voxel": asdict(point_cloud_stats(np.asarray(voxel_cloud.points))),
            "angle_area_voxel": asdict(point_cloud_stats(np.asarray(angle_cloud.points))),
        },
        "methods": {name: asdict(stats) for name, stats in methods.items()},
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark panorama-to-3D reconstruction methods.")
    parser.add_argument("--samples", nargs="+", default=["can", "pipe", "torus"])
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-hits", type=int, default=3)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--voxel-divisor", type=float, default=96.0)
    parser.add_argument("--include-exact-ray", action="store_true")
    parser.add_argument("--exact-ray-resolution", type=int, default=96)
    parser.add_argument("--include-subdivided-inversion", action="store_true")
    parser.add_argument("--subdivide-levels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="demo_outputs/reconstruction_benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sample_map = {
        "can": "demo_data/can/f82039689f504922995936c68484aa61.obj",
        "cow": "demo_data/cow_mesh/cow.obj",
        "pipe": "demo_data/pipe/fb42332b3f5e491cb0c4b5ba7ed6f374.obj",
        "torus": "demo_data/torus/torus.obj",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "config": {
            "samples": args.samples,
            "image_size": args.image_size,
            "max_hits": args.max_hits,
            "poisson_depth": args.poisson_depth,
            "voxel_divisor": args.voxel_divisor,
            "include_exact_ray": args.include_exact_ray,
            "exact_ray_resolution": args.exact_ray_resolution,
            "include_subdivided_inversion": args.include_subdivided_inversion,
            "subdivide_levels": args.subdivide_levels,
            "device": str(device),
        },
        "samples": {},
    }

    for sample_name in args.samples:
        if sample_name not in sample_map:
            raise ValueError(f"Unknown sample '{sample_name}'")
        print(f"Running sample: {sample_name}")
        result = run_sample(sample_name, sample_map[sample_name], args, device, args.output_dir)
        results["samples"][sample_name] = result

        best_method = min(
            result["methods"].items(),
            key=lambda item: item[1]["chamfer_mean"],
        )
        print(
            f"  best={best_method[0]} chamfer={best_method[1]['chamfer_mean']:.5f} "
            f"faces={best_method[1]['faces']} f1={best_method[1]['fscore_2pct']:.4f}"
        )

    result_path = os.path.join(args.output_dir, "results.json")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()