# Rectangular Conditional JiT Implementation Spec

## 1. Goal

Implement a conditional JiT-based model for custom training with the following requirements:

- Base backbone: **JiT**
- Image input resolution: **512×1024**
- Condition resolution: **512×1024**
- Image input channels: **1~3**
- Image output channels: **1~3**
- Condition types: **3 types**
- Condition channels: **up to ~35**
- Remove class-label conditioning
- Use a **global `g_token`** from the condition tower for global conditioning
- Support two switchable dense interaction modes:
  - `sparse_xattn`
  - `full_joint_mmdit`
- Do **not** use T2I-Adapter or an external vision tower; condition representations must be learned inside this model

Core design principle:

- Preserve as much of the pretrained JiT image backbone as possible
- Add condition-related modules as **inserted residual branches**
- Any newly added branch that injects into the **image stream** should be **zero-initialized** or near-zero-initialized

---

## 2. High-Level Design

### 2.1 Keep from JiT

Retain the following JiT components:

- `t_embedder`
- Image-side patch embedding design
- Image-side transformer backbone
- Image-side `adaLN-Zero`
- Image-side final prediction head
- JiT-style initialization philosophy for injected branches

### 2.2 Remove / Replace

Remove:

- `LabelEmbedder`
- Original class-based `y_emb`
- Original class-token / in-context-token insertion logic derived from `y_emb`

Replace with:

- A condition-tower-produced **global `g_token`**
- `g_proj(g_token)` as the global conditioning term
- A switchable dense interaction module

### 2.3 Final Global Conditioning

Define:

```math
c_{global} = t_{emb} + g_{emb}
```

```math
g_{emb} = \mathrm{MLP}(\mathrm{LN}(g\_token))
```

`c_global` is fed to **all image-side JiT blocks** and the **image-side final layer**.

This preserves the JiT/DiT global modulation pathway while replacing the original class embedding with a learned condition summary.

---

## 3. Step 1: Convert JiT to Rectangular Version

This is mandatory before adding conditioning.

### 3.1 Input Size as Tuple

All image-side and condition-side patch modules must support:

- `input_size = (H, W)`
- For this task:
  - `H = 512`
  - `W = 1024`

### 3.2 Patch / Grid Definition

Default:

- `patch_size = 32`
- `grid_h = H // patch_size = 16`
- `grid_w = W // patch_size = 32`
- `num_tokens = grid_h * grid_w = 512`

Optional higher-resolution token experiment:

- `patch_size = 16`
- `grid_h = 32`
- `grid_w = 64`
- `num_tokens = 2048`

**Default starting point: patch32**.

Reason:

- Both `sparse_xattn` and especially `full_joint_mmdit` become much more expensive with patch16
- The first working implementation should prioritize stability and tractable compute

### 3.3 Absolute Positional Embedding

Use for both image stream and condition stream:

- **Spatial tokens**: rectangular **fixed 2D sin-cos absolute positional embedding**
- **Global `g_token`**: **no 2D spatial PE**; only use a learned token embedding

### 3.4 RoPE

Use for both image stream and condition stream:

- **Rectangular 2D RoPE**
- Image spatial tokens: apply RoPE normally
- Condition spatial tokens: apply RoPE normally
- `g_token`: **do not apply spatial RoPE**

Implementation rule:

- Do not assign a fake 2D coordinate to `g_token`
- Apply rect-2D RoPE only to spatial tokens
- Treat `g_token` as a prefix token exempt from spatial rotation

### 3.5 Unpatchify

Must be changed to rectangular form.

Do **not** assume:

```math
h = w = \sqrt{T}
```

Instead explicitly use:

```math
h = grid_h,
\quad
w = grid_w
```

---

## 4. Image Input / Output Adapters

### 4.1 Image Input Adapter

To maximize JiT pretrained weight reuse, always convert image input to **3 channels** before the JiT image patch embedder.

Rules:

- If `Cin = 3`: `Identity`
- If `Cin != 3`: `Conv1x1(Cin -> 3)`

Suggested module name:

- `ImageInputAdapter`

### 4.2 Image Output Adapter

To avoid modifying the JiT final prediction head structure too aggressively, let JiT produce a **3-channel intermediate output**, then project to the target output channels.

Rules:

- If `Cout = 3`: `Identity`
- If `Cout != 3`: `Conv1x1(3 -> Cout)`

Suggested module name:

- `ImageOutputAdapter`

---

## 5. Condition Encoding Overview

Do **not** concatenate raw condition channels with the image input channels.

Use an independent condition path:

```math
\mathrm{cond}_{raw}
\rightarrow
\mathrm{CondTypeStem}
\rightarrow
\mathrm{CondPatchEmbed}
\rightarrow
[g\_token, C]
\rightarrow
\mathrm{CondTower}
```

Where:

- `g_token`: 1 global token
- `C`: spatial condition tokens

---

## 6. Condition Input Side

### 6.1 Three Condition Types

Do not use one fully shared raw stem for all condition types.

Create:

- `CondTypeStem[0]`
- `CondTypeStem[1]`
- `CondTypeStem[2]`

Each stem maps the corresponding raw condition channels to a unified intermediate width `cond_base_channels`.

Recommended per-type stem structure:

- `Conv3x3 -> SiLU -> Conv3x3 -> SiLU -> Conv1x1`

Suggested output width:

- `cond_base_channels = 8` or `16`

Then all condition types share:

- `CondPatchEmbedRect`
- `CondTower`
- `g_proj`

This allows different low-level channel statistics to be normalized before entering a shared token space.

---

## 7. Condition Patch Embedding

Create:

- `CondPatchEmbedRect(input_size=(512,1024), patch_size=p, in_chans=cond_base_channels, bottleneck_dim=b_cond, embed_dim=D)`

Recommended implementation style:

- Follow JiT `BottleneckPatchEmbed`
- Use one large-stride patchifying conv
- Then a `1×1` conv or equivalent linear projection to `hidden_size = D`

This keeps image and condition streams aligned at the patch-token level.

---

## 8. Condition Tower

### 8.1 Token Composition

The condition tower input sequence is:

```math
[g\_token, C_1, \dots, C_N]
```

Where:

- `g_token` is a learned parameter with shape `[1, 1, D]`
- `C_i` are patchified condition spatial tokens

#### Positional Encoding Rules

- `g_token`: learned token embedding only; **no 2D spatial PE**
- `C`: add rectangular fixed 2D sin-cos PE

#### RoPE Rules

- `g_token`: no RoPE
- `C`: rect-2D RoPE

### 8.2 Tower Depth

Default recommendation:

- `L_cond = 4` for patch32
- `L_cond = 6` for patch16

Do **not** make this tower too shallow.

Its role is not just patchification, but also building a dedicated semantic token space for the condition.

### 8.3 Tower Block Form

Use JiT/DiT-style transformer blocks for the condition tower, but condition them only on `t_emb`, not on `g_emb`.

Reason:

- The condition tower should be timestep-aware
- But `g_emb` depends on the final `g_token`, so using `g_emb` inside the tower would introduce circular dependency

Each `CondBlock` should include:

- Self-attention
- MLP
- Rect-2D RoPE
- Either `adaLN-Zero` or a lighter modulated-LN using `t_emb`

### 8.4 Tower Output

The final outputs are:

- `g_token_final`
- `C_final`

Then compute:

```math
g_{emb} = \mathrm{MLP}(\mathrm{LN}(g\_token_{final}))
```

```math
c_{global} = t_{emb} + g_{emb}
```

---

## 9. Image Stream

The image stream remains JiT-based:

```math
x
\rightarrow
\mathrm{ImageInputAdapter}
\rightarrow
\mathrm{JiT\ PatchEmbed}
\rightarrow
X
```

```math
X \leftarrow X + \mathrm{PE}_{img}
```

```math
X \leftarrow \mathrm{JiT\ blocks}(X; c_{global})
```

Differences from original JiT:

- No `LabelEmbedder`
- No class-based in-context token insertion
- Additional interaction blocks inserted at selected layers

---

## 10. Dense Interaction Modes

Add a config option:

- `interaction_mode = "sparse_xattn"`
- `interaction_mode = "full_joint_mmdit"`

Both modes share:

- The same `CondTower`
- The same `g_token` / `g_proj`
- The same image JiT backbone
- The same `interaction_layers`

Recommended default insertion layers:

- For a 12-layer backbone: `[2, 5, 8, 11]`

These interaction modules must be **inserted after standard JiT image blocks**, not used to replace the JiT blocks themselves.

---

## 11. Mode A: `sparse_xattn`

This is the default and must be implemented first.

### 11.1 Structure

At each `interaction_layer`, insert:

- `SparseCrossAttnAdapter`

Inputs:

- query: `X` (image tokens)
- key/value: `[g_token_final, C_final]` (condition tokens)

Update rule:

```math
X \leftarrow X + \alpha_i \cdot \mathrm{XAttn}(\mathrm{LN}_q(X),\ \mathrm{LN}_{kv}([g,C]),\ \mathrm{LN}_{kv}([g,C]))
```

Where:

- `alpha_i` is one learnable scalar per interaction block
- Initialize `alpha_i = 0`
- The adapter `out_proj` should also be zero-initialized if practical

### 11.2 PE / RoPE Handling

- Query-side image tokens: image rect-2D RoPE
- KV-side condition spatial tokens: condition rect-2D RoPE
- `g_token`: no RoPE

Implementation rule:

- Apply RoPE separately to image and condition streams before attention
- Do not force `g_token` into a 2D coordinate grid

### 11.3 Condition Token Update Policy

In `sparse_xattn` mode:

- Do **not** update `C_final`
- Compute the condition tower once
- Reuse the same `[g_token_final, C_final]` for all sparse interaction blocks

This reduces complexity and stabilizes the first version.

---

## 12. Mode B: `full_joint_mmdit`

This is the stronger but heavier option.

### 12.1 Core Idea

Do **not** replace the JiT main blocks.

Instead, at each `interaction_layer`, insert a **dual-stream MMDiT-style residual interaction block**.

Inputs:

- image stream: `X`
- condition stream: `[g, C]`

Inside the block:

- Image and condition each have their own norm / qkv / out_proj / MLP parameters
- Attention is computed jointly across the concatenated token set
- Outputs are split back into image and condition streams

### 12.2 Block Definition

Create:

- `FullJointMMDiTAdapter`

#### Attention Part

First compute modality-specific projections:

```math
Q_x, K_x, V_x = W^x_{qkv}(\mathrm{LN}_x(X))
```

```math
Q_c, K_c, V_c = W^c_{qkv}(\mathrm{LN}_c([g,C]))
```

Concatenate:

```math
Q = [Q_c; Q_x],
\quad
K = [K_c; K_x],
\quad
V = [V_c; V_x]
```

Compute joint attention:

```math
O = \mathrm{Attn}(Q, K, V)
```

Split back:

```math
O = [O_c; O_x]
```

Project separately:

```math
\Delta C = W^c_o(O_c)
```

```math
\Delta X = W^x_o(O_x)
```

Update:

```math
[g,C] \leftarrow [g,C] + \Delta C
```

```math
X \leftarrow X + \alpha_i \cdot \Delta X
```

Where:

- `alpha_i` is learnable and initialized to `0`

#### MLP Part

Then do modality-specific MLP updates:

```math
[g,C] \leftarrow [g,C] + \mathrm{MLP}_c(\mathrm{LN}_c([g,C]))
```

```math
X \leftarrow X + \beta_i \cdot \mathrm{MLP}_x(\mathrm{LN}_x(X))
```

Where:

- `beta_i` is learnable and initialized to `0`

### 12.3 Why the Zero Gates

This block is an **inserted multimodal residual adapter**, not the pretrained backbone itself.

Therefore:

- Any branch injecting residual updates into the **image stream** must be zero-initialized or gated by zero-initialized scalars
- The **condition-stream-only** residual path does not need zero-init in the same strict sense

### 12.4 Condition Token Update Policy

In `full_joint_mmdit` mode:

- Update `[g, C]` inside each joint block
- Later joint blocks should use the updated condition stream

This allows progressive alignment between image and condition streams.

### 12.5 PE / RoPE Handling

- Image spatial tokens: image rect-2D RoPE
- Condition spatial tokens: condition rect-2D RoPE
- `g_token`: no RoPE

Implementation rule:

- Apply RoPE within each modality first
- Then concatenate for joint attention
- Do **not** use one mixed-grid RoPE after concatenation

---

## 13. Implementation Priority

Recommended execution order:

1. Implement and validate `sparse_xattn`
2. Then implement `full_joint_mmdit`

Reason:

- `sparse_xattn` is simpler and safer
- `full_joint_mmdit` is more powerful but heavier and easier to destabilize

---

## 14. Weight Loading Strategy

### 14.1 Load from Pretrained JiT

Load pretrained weights into:

- Image-side `x_embedder`
- `t_embedder`
- All image-side JiT blocks
- Image-side final layer

Conditions:

- The image stream remains 3-channel before patch embedding
- Patch size matches the chosen JiT pretrained checkpoint
- Hidden size / depth / heads match the selected JiT base model

### 14.2 Randomly Initialize

Random initialization for:

- `ImageInputAdapter` if `Cin != 3`
- `ImageOutputAdapter` if `Cout != 3`
- All `CondTypeStem`
- `CondPatchEmbedRect`
- `g_token`
- `CondTower`
- `g_proj`
- All interaction adapters / blocks

### 14.3 Initialization Rules

Follow JiT/DiT zero-init philosophy:

- Any newly added residual path that writes into the **image stream**:
  - set `alpha_i = 0`
  - set `beta_i = 0` where applicable
  - zero-init `out_proj` if practical
- The last layer of `g_proj` can use small initialization
- The condition tower can use standard Xavier / truncated normal

---

## 15. Training Schedule

### 15.1 Multi-Phase Training

#### Phase 1

Freeze:

- All original image-side JiT backbone blocks
- Optionally freeze the original image-side final layer

Train only:

- `ImageInputAdapter`
- `ImageOutputAdapter`
- `CondTypeStem`
- `CondPatchEmbedRect`
- `CondTower`
- `g_proj`
- Interaction blocks

Purpose:

- Let the condition path learn to enter JiT feature space first
- Reduce risk of damaging the pretrained backbone early

#### Phase 2

Unfreeze:

- Image-side final layer
- Last 1/3 of the image backbone blocks

Learning rate:

- backbone LR = 0.1 ~ 0.2 × new-module LR

#### Phase 3 (optional)

Unfreeze all:

- Full backbone with small LR for joint finetuning

### 15.2 Recommended Experiment Order

Run experiments in this order:

1. `patch32 + sparse_xattn`
2. `patch32 + full_joint_mmdit`
3. `patch16 + sparse_xattn`
4. `patch16 + full_joint_mmdit`

Do **not** start with `patch16 + full_joint_mmdit`.

---

## 16. Recommended Default Hyperparameters

For the first implementation:

- backbone base: **JiT-B/32**-compatible configuration
- image resolution: `512×1024`
- condition resolution: `512×1024`
- patch size: `32`
- `interaction_mode`: `sparse_xattn`
- `interaction_layers`: `[2, 5, 8, 11]`
- `L_cond = 4`
- `cond_base_channels = 16`
- `g_token_count = 1`

Then add:

- `interaction_mode = full_joint_mmdit`

---

## 17. Forward Pass Specification

### Step 1: Image Path

```math
x_{raw}
\rightarrow
\mathrm{ImageInputAdapter}
\rightarrow
x_{rgb}
```

```math
X = \mathrm{ImagePatchEmbed}(x_{rgb}) + \mathrm{PE}_{img}
```

### Step 2: Condition Path

```math
cond_{raw}
\rightarrow
\mathrm{CondTypeStem}[type\_id]
\rightarrow
cond_{base}
```

```math
C = \mathrm{CondPatchEmbed}(cond_{base}) + \mathrm{PE}_{cond}
```

```math
[g,C] = \mathrm{CondTower}([g,C]; t_{emb})
```

### Step 3: Global Conditioning

```math
g_{emb} = \mathrm{MLP}(\mathrm{LN}(g))
```

```math
c_{global} = t_{emb} + g_{emb}
```

### Step 4: Image Backbone + Interactions

For each image block `i`:

```math
X = \mathrm{JiTBlock}_i(X; c_{global})
```

If `i in interaction_layers`:

- `sparse_xattn`:

```math
X = \mathrm{SparseCrossAttnAdapter}_i(X, [g,C])
```

- `full_joint_mmdit`:

```math
X, [g,C] = \mathrm{FullJointMMDiTAdapter}_i(X, [g,C], c_{global})
```

### Step 5: Output Path

```math
Y_{3ch} = \mathrm{JiTFinalLayer}(X; c_{global})
```

```math
Y = \mathrm{ImageOutputAdapter}(Y_{3ch})
```

---

## 18. Explicitly Forbidden Implementations

Do **not** do the following:

1. Do **not** concatenate raw condition channels with raw image channels and feed them directly into JiT
2. Do **not** remove the entire `adaLN` system
3. Do **not** treat `g_token` as a spatial token with 2D RoPE
4. Do **not** share the same qkv / MLP parameters between image and condition in `full_joint_mmdit`
5. Do **not** replace the main JiT backbone blocks with joint blocks; interaction modules must be **inserted residual adapters**
6. Do **not** start with `patch16 + full_joint_mmdit`

---

## 19. Acceptance Checklist

### 19.1 Shape Checks

For patch32:

- Image token shape: `B × 512 × D`
- Condition token shape: `B × (1 + 512) × D`
- Output shape: `B × Cout × 512 × 1024`

### 19.2 PE / RoPE Checks

Verify:

- Image spatial tokens use rect-2D absolute PE + rect-2D RoPE
- Condition spatial tokens use rect-2D absolute PE + rect-2D RoPE
- `g_token` uses no spatial PE and no RoPE

### 19.3 Initialization Checks

Verify:

- All image-side interaction adapter output gates are initialized to `0`
- Initial forward pass with condition enabled behaves close to base JiT with minimal injected effect

### 19.4 Mode Switch Checks

Ensure the same backbone can switch modes via config only:

- `interaction_mode="sparse_xattn"`
- `interaction_mode="full_joint_mmdit"`

---

## 20. Minimal First Milestone

The first milestone that must work end-to-end is:

**Rectangular JiT-B/32 + CondTower(4 blocks) + `g_token -> g_proj -> adaLN` + `sparse_xattn` adapters at `[2,5,8,11]` + zero-init image-side adapter residuals.**

Only after this is stable should the implementation add:

**`full_joint_mmdit` interaction blocks**, still as inserted adapters rather than replacements of the main JiT blocks.
