import torch
import numpy as np
import torch.nn.functional as F
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer.mesh.shader import phong_shading, ShaderBase

def flip_cubemap_to_fit(cube_images):
    '''
    The following setting is for camera orientation:
    camera_dirs = torch.tensor([
        [-1, 0, 0],   # -X
        [1, 0, 0],    # +X
        [0, -1, 0],   # -Y
        [0, 1, 0],    # +Y
        [0, 0, -1],   # -Z
        [0, 0, 1],    # +Z
    ], device=device, dtype=torch.float32)

    # Custom up vectors to avoid singularities
    up_vectors = torch.tensor([
        [0, 1, 0],    # for -X
        [0, 1, 0],    # for +X
        [0, 0, 1],   # for -Y
        [0, 0, 1],    # for +Y (must not be parallel to +Y dir)
        [0, 1, 0],    # for -Z
        [0, 1, 0],    # for +Z
    ], device=device, dtype=torch.float32)
    '''
    cube_images[0] = torch.flip(cube_images[0], dims=(1,))
    cube_images[2] = torch.flip(cube_images[2], dims=(1,))
    cube_images[4] = torch.flip(cube_images[4], dims=(1, 2))
    cube_images[5] = torch.flip(cube_images[5], dims=(2,))
    return cube_images

def cubemap_to_equirectangular(cube_faces, H, device="cpu"):
    """
    Convert 6 cube face images into a single equirectangular panorama.

    Args:
        cube_faces: (6, C, F, F) tensor of cube face images
                    Order: [-X, +X, -Y, +Y, -Z, +Z]
        H: height of the panorama image (output will be [C, H, 2H])
        device: CPU or CUDA device

    Returns:
        pano_img: (C, H, 2H) panorama image
    """
    cube_faces = cube_faces.to(device)
    C, f = cube_faces.shape[1], cube_faces.shape[2]
    W = 2 * H

    # Create spherical coordinates
    phi = torch.linspace(0, np.pi, H, device=device).view(H, 1)        # [0, pi]
    theta = torch.linspace(-np.pi, np.pi, W, device=device).view(1, W) # [-pi, pi]
    x = torch.sin(phi) * torch.sin(theta)
    y = torch.cos(phi).repeat(1, W)
    z = torch.sin(phi) * torch.cos(theta)
    dirs = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    # Normalize and map to cube face
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    abs_dirs = dirs.abs()
    max_axis = abs_dirs.argmax(dim=-1)

    uv = torch.empty(H, W, 2, device=device)
    face_idx = torch.empty(H, W, dtype=torch.long, device=device)

    def project(face, major, u, v):
        idx = (max_axis == major) & (dirs[..., major] * face > 0)
        denom = dirs[..., major][idx].unsqueeze(-1)
        uv_ = dirs[idx][..., [u, v]] / denom
        uv_ = (uv_ + 1) / 2  # normalize to [0, 1]
        uv[idx] = uv_
        face_idx[idx] = {
            (0, 1): 0,  # +X
            (0, -1): 1, # -X
            (1, 1): 2,  # +Y
            (1, -1): 3, # -Y
            (2, 1): 4,  # +Z
            (2, -1): 5, # -Z
        }[(major, face)]

    # Project each face
    project(1, 0, 2, 1)   # +X
    project(-1, 0, 2, 1)  # -X
    project(1, 1, 0, 2)   # +Y
    project(-1, 1, 0, 2)  # -Y
    project(1, 2, 0, 1)   # +Z
    project(-1, 2, 0, 1)  # -Z

    # Rescale UV to [-1, 1] for grid_sample
    uv = 2 * uv - 1
    uv = uv.view(1, H, W, 2)

    pano = torch.zeros(C, H, W, device=device)

    for i in range(6):
        face_img = cube_faces[i].unsqueeze(0)  # (1, C, f, f)
        mask = (face_idx == i)
        if not mask.any():
            continue
        sample_uv = uv.clone()
        sample_uv[0][~mask] = -10  # force out-of-bounds
        # sampled = F.grid_sample(face_img, sample_uv, mode='bilinear', align_corners=True)
        sampled = F.grid_sample(face_img, sample_uv, mode='nearest', align_corners=True)
        pano[:, mask] = sampled[0, :, mask]

    return pano

def pano_to_pointcloud(pano_color, pano_dist, valid_mask=None):
    """
    Convert equirectangular panorama + distance map into a 3D point cloud
    Args:
        pano_color: (3, H, 2H) RGB tensor
        pano_dist: (H, 2H) distance-to-origin tensor
    Returns:
        points_world: (N, 3) 3D points
        colors: (N, 3) RGB colors
    """
    C, H, W = pano_color.shape
    assert W == 2 * H, "Equirectangular image must have width = 2 * height"

    device = pano_color.device
    pano_color = pano_color.permute(1, 2, 0)  # (H, W, 3)
    pano_dist = pano_dist  # (H, W)

    # Generate spherical angles
    phi = torch.linspace(0, np.pi, H, device=device).view(H, 1)        # vertical (0 to pi)
    theta = torch.linspace(-np.pi, np.pi, W, device=device).view(1, W) # horizontal (-pi to pi)

    # Compute unit direction vectors (camera rays)
    x = torch.sin(phi) * torch.sin(theta)
    y = torch.cos(phi).repeat(1, W)
    z = torch.sin(phi) * torch.cos(theta)
    dirs = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    points = dirs * pano_dist.unsqueeze(-1)  # (H, W, 3)
    colors = pano_color  # (H, W, 3)

    # Flatten
    points_flat = points.view(-1, 3)
    colors_flat = colors.view(-1, 3)

    if valid_mask is not None:
        valid_mask = valid_mask.view(-1)
        points_flat = points_flat[valid_mask]
        colors_flat = colors_flat[valid_mask]
        pano_dist = pano_dist.view(-1)[valid_mask]

    # Optional: remove invalid distances (e.g., 0 or nan)
    valid = torch.isfinite(points_flat).all(dim=1) & (pano_dist.view(-1) > 1e-5)
    points_world = points_flat[valid]
    colors_world = colors_flat[valid]

    return points_world, colors_world

def get_sampling_prob_from_normal_angle(normal_angle_map, max_prob=1.0, min_prob=0.1, max_thres=0.866, min_thres=0.5):
    """
    Compute sampling probability based on normal angle map.
    """
    sampling_prob = torch.zeros_like(normal_angle_map)
    sampling_prob = torch.where(normal_angle_map > max_thres, min_prob, sampling_prob)
    sampling_prob = torch.where(normal_angle_map < min_thres, max_prob, sampling_prob)
    sampling_prob = torch.where(
        (normal_angle_map >= min_thres) & (normal_angle_map <= max_thres), 
        max_prob + (normal_angle_map - min_thres) * (max_prob - min_prob) / (min_thres - max_thres),
        sampling_prob
    )

    return sampling_prob

def pano_to_mesh(pano_color, pano_dist, valid_mask=None, max_edge_ratio=1.5):
    """
    Convert a panorama + distance map into a mesh via 3D triangulation.

    Args:
        pano_color: (3, H, 2H) RGB image in equirectangular projection
        pano_dist:  (H, 2H)   Distance-to-origin (Euclidean, not depth)
        valid_mask: (H, 2H)   Optional binary mask for valid pixels
        max_edge_ratio: float; skip triangles where any edge is longer than
                        max_edge_ratio * median edge length (for outlier rejection)

    Returns:
        verts: (N, 3) torch.FloatTensor
        faces: (M, 3) torch.LongTensor
        colors: (N, 3) torch.FloatTensor
    """

    C, H, W = pano_color.shape
    device = pano_color.device
    pano_color = pano_color.permute(1, 2, 0)  # (H, W, 3)

    # Step 1: spherical directions
    phi = torch.linspace(0, np.pi, H, device=device).view(H, 1)
    theta = torch.linspace(-np.pi, np.pi, W, device=device).view(1, W)
    x = torch.sin(phi) * torch.sin(theta)
    y = torch.cos(phi).repeat(1, W)
    z = torch.sin(phi) * torch.cos(theta)
    dirs = torch.stack([x, y, z], dim=-1)  # (H, W, 3)
    pts = dirs * pano_dist.unsqueeze(-1)  # (H, W, 3)

    if valid_mask is None:
        valid_mask = torch.isfinite(pano_dist) & (pano_dist > 1e-5)

    # Step 2: assign vertex index grid
    index_map = -torch.ones((H, W), dtype=torch.long, device=device)
    valid_idx = valid_mask.nonzero(as_tuple=False)
    index_map[valid_mask] = torch.arange(valid_idx.shape[0], device=device)

    verts = pts[valid_mask]        # (N, 3)
    colors = pano_color[valid_mask]  # (N, 3)

    # Step 3: triangle construction (grid-based)
    faces = []
    y = torch.arange(H - 1, device=device)
    x = torch.arange(W, device=device) # set to W to wrap around
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    i00 = index_map[yy, xx]
    i01 = index_map[yy, (xx + 1) % W]
    i10 = index_map[yy + 1, xx]
    i11 = index_map[yy + 1, (xx + 1) % W]

    # Two triangles per quad (valid if all indices >= 0)
    mask_a = (i00 >= 0) & (i01 >= 0) & (i10 >= 0)
    mask_b = (i11 >= 0) & (i01 >= 0) & (i10 >= 0)

    tri_a = torch.stack([i00[mask_a], i01[mask_a], i10[mask_a]], dim=-1)
    tri_b = torch.stack([i11[mask_b], i01[mask_b], i10[mask_b]], dim=-1)
    faces.append(tri_a)
    faces.append(tri_b)

    faces = torch.cat(faces, dim=0)  # (M, 3)

    # Step 4: optional filtering based on edge lengths
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1 = (v0 - v1).norm(dim=1)
    e2 = (v1 - v2).norm(dim=1)
    e3 = (v2 - v0).norm(dim=1)
    edge_lengths = torch.cat([e1, e2, e3])
    edge_median = edge_lengths.median()
    valid_tri = (e1 < edge_median * max_edge_ratio) & \
                (e2 < edge_median * max_edge_ratio) & \
                (e3 < edge_median * max_edge_ratio)

    faces = faces[valid_tri]

    return verts, faces, colors


class DistShader(torch.nn.Module):
    def __init__(self, cameras, max_hits=3):
        super().__init__()
        self.cameras = cameras
        self.max_hits = max_hits

    """
    Shader that returns per-pixel Euclidean distance to the camera origin.
    """
    def forward(self, fragments, meshes, **kwargs):
        pix_to_face = fragments.pix_to_face  # (N, H, W, K)
        bary_coords = fragments.bary_coords  # (N, H, W, K, 3)

        verts = meshes.verts_packed()        # (V, 3)
        faces = meshes.faces_packed()        # (F, 3)
        faces_verts = verts[faces]           # (F, 3, 3)

        # Get face vertices for each pixel using advanced indexing
        N, H, W, K = pix_to_face.shape
        pixel_faces = pix_to_face.view(N, -1, K)  # (N, H*W, K)
        pixel_bary = bary_coords.view(N, -1, K, 3)  # (N, H*W, K, 3)

        # Get the vertex positions
        pixel_verts = faces_verts[pixel_faces]  # (N, H*W, K, 3, 3)

        # Barycentric interpolation to get 3D points
        bary = pixel_bary.unsqueeze(-1)  # (N, H*W, K, 3, 1)
        pts = (pixel_verts * bary).sum(dim=-2)  # (N, H*W, K, 3)

        # Use the closest sample (first face hit)
        all_dists = []
        for i in range(self.max_hits):
            this_pts = pts[:, :, i, :]
            dist = this_pts.norm(dim=-1)
            dist = dist.view(N, H, W, 1)
            all_dists.append(dist)

        return all_dists # num_hits x (N, H, W, 1)

class NormalAngleShader(torch.nn.Module):
    def __init__(self, cameras, max_hits=3):
        super().__init__()
        self.cameras = cameras
        self.max_hits = max_hits

    """
    Shader that returns per-pixel dot product between face normal and view vector.
    """
    def forward(self, fragments, meshes, **kwargs):
        pix_to_face = fragments.pix_to_face  # (N, H, W, K)
        bary_coords = fragments.bary_coords  # (N, H, W, K, 3)

        verts = meshes.verts_packed()        # (V, 3)
        faces = meshes.faces_packed()        # (F, 3)
        faces_verts = verts[faces]           # (F, 3, 3)
        face_normals = torch.cross(
            faces_verts[:, 1] - faces_verts[:, 0],
            faces_verts[:, 2] - faces_verts[:, 0],
            dim=-1
        )  # (F, 3)
        face_normals = torch.nn.functional.normalize(face_normals, dim=-1)

        # Camera origin
        cam_origin = self.cameras.get_camera_center().unsqueeze(1).unsqueeze(1)  # (N, 1, 1, 3)

        # Prepare geometry
        N, H, W, K = pix_to_face.shape
        pixel_faces = pix_to_face.view(N, -1, K)            # (N, H*W, K)
        pixel_bary = bary_coords.view(N, -1, K, 3)          # (N, H*W, K, 3)
        pixel_verts = faces_verts[pixel_faces]              # (N, H*W, K, 3, 3)

        bary = pixel_bary.unsqueeze(-1)                     # (N, H*W, K, 3, 1)
        pts = (pixel_verts * bary).sum(dim=-2)              # (N, H*W, K, 3)

        pixel_normals = face_normals[pixel_faces]           # (N, H*W, K, 3)

        view_vec = torch.nn.functional.normalize(pts - cam_origin, dim=-1)  # (N, H*W, K, 3)

        all_dot = []
        for i in range(self.max_hits):
            normal = pixel_normals[:, :, i, :]
            view = view_vec[:, :, i, :]
            dot = (normal * view).sum(dim=-1)  # (N, H*W)
            dot = dot.view(N, H, W, 1)
            all_dot.append(dot)

        return all_dot  # num_hits x (N, H, W, 1)

class MultiHitHardPhongShader(ShaderBase):
    def __init__(self, max_hits=4, **kwargs):
        super().__init__(**kwargs)
        self.max_hits = max_hits

    def forward(self, fragments: Fragments, meshes, **kwargs):
        # Modify rasterizer to return more faces per pixel (top k instead of only nearest)
        # By default, only the closest face is returned. To get top-k:
        # fragments.pix_to_face: (N, H, W, K)
        # fragments.zbuf: (N, H, W, K)

        cameras = super()._get_cameras(**kwargs)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)

        # Example: skip first hit (k=1), use second and beyond
        # Here we process all k hit faces
        pix_to_face = fragments.pix_to_face  # (N, H, W, K)
        bary_coords = fragments.bary_coords
        zbuf = fragments.zbuf

        all_images = []
        for idx in range(self.max_hits):
            this_fragments = Fragments(
                pix_to_face=pix_to_face[..., idx:idx+1],
                zbuf=zbuf[..., idx:idx+1],
                bary_coords=bary_coords[..., idx:idx+1, :],
                dists=fragments.dists[..., idx:idx+1],
            )
            texels = meshes.sample_textures(this_fragments)

            colors = phong_shading(
                meshes=meshes,
                fragments=this_fragments,
                texels=texels,
                lights=lights,
                cameras=cameras,
                materials=materials,
            )
            images = hard_rgb_blend(
                colors, this_fragments, blend_params
            )
            all_images.append(images)
        return all_images