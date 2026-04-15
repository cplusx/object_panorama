# Inverse-Spherical Representation and Adaptive Subdivision

## 1. Problem Statement

The target is not generic surface reconstruction. The target is a unified, image-like 3D representation that can be used for learning. In this repository, that representation is a layered spherical field indexed by viewing direction.

Let the object be a closed or partially open surface $\mathcal{M} \subset \mathbb{R}^3$, and let the representation origin be the point $o = 0$. For a unit direction $\omega \in S^2$, define the inward ray

$$
r_{\omega}(t) = R\omega - t\omega, \quad t \ge 0,
$$

where $R$ is an outer sphere radius chosen so that the object lies inside the sphere. For this ray, let the ordered intersections with the mesh be

$$
x_0(\omega), x_1(\omega), \dots, x_{K-1}(\omega),
$$

sorted by increasing ray depth. The layered inverse-spherical representation is then

$$
\mathcal{S}(\omega) = \left\{\rho_k(\omega), c_k(\omega), m_k(\omega)\right\}_{k=0}^{K-1},
$$

with

$$
\rho_k(\omega) = \frac{1}{\|x_k(\omega)\|},
$$

where $c_k(\omega)$ is color, and $m_k(\omega)$ is a validity mask. In tensor form, this becomes a stack of image-like maps over an angular parameterization of $S^2$.

This is the correct object to optimize for. It is already a canonical training representation.

## 2. What the Previous Pipeline Approximated

The previous pipeline attempted to obtain the same representation indirectly:

1. Normalize the mesh.
2. Apply pointwise spherical inversion to mesh vertices,

$$
I(x) = \frac{x}{\|x\|^2}.
$$

3. Reuse the original triangle connectivity on the inverted vertices.
4. Rasterize the inverted mesh with PyTorch3D.
5. Interpret rasterized distances as the desired spherical representation.

This is attractive because it converts the problem into ordinary mesh rendering. However, it contains a geometric inconsistency.

## 3. Why Vertex-Wise Inversion plus Triangle Rasterization is Inconsistent

Spherical inversion is not an affine map. Therefore, the image of a planar triangle is generally not another planar triangle.

More precisely:

- A line or plane not passing through the inversion center maps to a circle or sphere passing through the center.
- A triangle on a plane not passing through the center maps to a curved patch on a sphere-like surface.
- Replacing that curved patch by the triangle formed by the inverted vertices introduces approximation error.

So if a triangle $T$ has vertices $v_0, v_1, v_2$, then the true inverted patch is

$$
I(T) = \{ I(x) : x \in T \},
$$

but the rasterizer actually uses the planar surrogate

$$
\tilde{T} = \operatorname{conv}(I(v_0), I(v_1), I(v_2)).
$$

For triangles far from the origin and with small spatial extent, $I(T)$ and $\tilde{T}$ are close. For triangles with large extent or small radius, the discrepancy becomes severe.

This is exactly why `can` behaved well while `pipe` failed.

## 4. Why Pipe Fails Specifically

For `pipe`, the mesh contains regions that are both:

- geometrically elongated,
- and relatively close to the inversion center.

The inversion map has Jacobian magnitude proportional to $1 / \|x\|^2$, and its local curvature distortion grows rapidly as $\|x\|$ decreases. Therefore, coarse triangles near the origin become highly curved after inversion. Replacing those curved patches by single planar triangles creates a large geometric bias before any reconstruction or meshing step even begins.

Empirically, this was confirmed by two observations:

1. The inversion-derived point set for `pipe` had a recovered spatial scale far larger than the source mesh scale.
2. Direct inward ray intersections on the original mesh produced a much better layered spherical field than the inverted-mesh rasterization.

Therefore, the main failure is in the inversion approximation, not in the later Poisson or BPA meshing.

## 5. Adaptive Subdivision: What It Means

Adaptive subdivision is not a new representation. It is an approximation strategy for rendering the same representation more accurately.

The idea is to subdivide triangles more heavily where inversion is more nonlinear, and less where inversion is approximately linear.

Let $T$ be a triangle with vertices $v_0, v_1, v_2$. A principled inversion error indicator is

$$
\delta(T) = \max_{x \in \mathcal{Q}(T)} \operatorname{dist}\left(I(x), \Pi_T\right),
$$

where:

- $\mathcal{Q}(T)$ is a set of probe points such as edge midpoints and the triangle centroid,
- $\Pi_T$ is the plane through $I(v_0), I(v_1), I(v_2)$.

In practice, one can use midpoint tests:

$$
\delta_{ij} = \left\| I\left(\frac{v_i + v_j}{2}\right) - \frac{I(v_i) + I(v_j)}{2} \right\|,
$$

and subdivide whenever

$$
\max(\delta_{01}, \delta_{12}, \delta_{20}) > \varepsilon.
$$

This is the correct interpretation of adaptive subdivision: refine precisely where inversion bends the surface too much for a single triangle to approximate it.

### Uniform vs Adaptive Subdivision

- Uniform subdivision splits every triangle to the same depth.
- Adaptive subdivision splits only triangles with large inversion error.

Uniform subdivision already improved `pipe` significantly in our tests. Adaptive subdivision should be better because it concentrates triangles exactly where the inversion distortion is largest.

## 6. Exact Ray vs Adaptive Subdivision

These two methods solve different problems:

### Exact ray

Exact ray computes the representation directly from the original mesh by intersecting inward spherical rays with the original triangles. No inverted mesh is ever built. This yields the most faithful inverse-spherical field.

It answers the question:

> What is the correct layered spherical representation of this mesh?

### Adaptive subdivision

Adaptive subdivision keeps the old rendering pipeline, but makes its geometric approximation less wrong by refining triangles before inversion.

It answers the question:

> If I insist on using inverted-mesh rasterization, how can I reduce the error?

Therefore:

- Exact ray is a direct construction of the desired representation.
- Adaptive subdivision is an approximation scheme for a legacy rendering trick.

## 7. Which Option Gives a Unified Training Representation?

Yes, the goal of a unified, image-like 3D representation is achievable.

The cleanest option is a layered spherical tensor. For resolution $H \times 2H$ and $K$ hits, define a tensor

$$
\mathbf{X} \in \mathbb{R}^{K \times C \times H \times 2H},
$$

where the channel set can be

- inverse radius $\rho$,
- raw radius $r$,
- RGB color,
- valid mask,
- optional normal or confidence channels.

A practical training tensor could be:

$$
[\rho, r, R, G, B, m] \in \mathbb{R}^{K \times 6 \times H \times 2H}.
$$

This satisfies the uniform-representation requirement across all objects.

### Recommended representations for training

1. Layered equirectangular inverse-spherical maps.
Reason: easiest to integrate with image models and current code.

2. Layered cubemap inverse-spherical maps.
Reason: less polar distortion, better geometric uniformity than equirectangular.

3. Optional equal-area spherical parameterization.
Reason: best sampling uniformity, but more engineering overhead.

For this repository, the first option is the most natural immediate target.

## 8. Recommended Direction for This Repository

If the end goal is a learnable 3D representation, the recommended pipeline is:

1. Use inward rays on the original mesh.
2. Record the ordered hit radii and colors on a fixed angular grid.
3. Store the result as a layered spherical tensor.
4. Treat meshing only as an optional visualization or evaluation step.

In other words, the representation should be primary, and the mesh should be secondary.

This is why the newly added `inverse_spherical_representation.py` module is the right conceptual direction: it computes the layered spherical field directly, instead of relying on the invalid assumption that inversion preserves triangle planarity.

## 9. Practical Consequences

If the priority is correctness:

- prefer direct inward ray tracing on the original mesh.

If the priority is speed while keeping the current rasterization design:

- use adaptive subdivision before inversion,
- and refine based on inversion error rather than a fixed global subdivision depth.

If the priority is training a model on a unified representation:

- use a layered spherical tensor representation,
- not a mesh or a point cloud as the primary supervision target.

## 10. Files Added for This Work

- `inverse_spherical_representation.py`
  Direct construction of the layered inverse-spherical representation from ray intersections.

- `reconstruction_benchmark.py`
  Diagnostic and comparison code used to verify that the failure on `pipe` was caused by the inversion approximation rather than by meshing alone.

- `docs/inspect_pipe_outputs.ipynb`
  Notebook to inspect the generated `ply` files and the corresponding metrics visually.