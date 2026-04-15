from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from torch import nn

from .jit_checkpoint_loader import (
    instantiate_model_from_public_checkpoint,
    load_public_weights_into_model,
)
from .jit_layers import (
    BottleneckPatchEmbed,
    FinalLayer,
    JiTBlock,
    RMSNorm,
    SwiGLUFFN,
    TimestepEmbedder,
    get_2d_sincos_pos_embed_from_grid,
    modulate,
    rotate_half,
    scaled_dot_product_attention,
)
from .jit_model import JIT_PRESET_CONFIGS, JiTModel, create_jit_model


def _to_2tuple(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple, got {value!r}")
        return int(value[0]), int(value[1])
    size = int(value)
    return size, size


def _default_interaction_layers(depth: int) -> tuple[int, ...]:
    if depth <= 0:
        return ()
    num_points = min(4, depth)
    layers = []
    for index in range(num_points):
        layer = round(depth * (index + 1) / num_points) - 1
        layers.append(max(0, min(depth - 1, layer)))
    return tuple(dict.fromkeys(layers))


def _validate_interaction_mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized not in {"sparse_xattn", "full_joint_mmdit"}:
        raise ValueError(f"Unsupported interaction_mode '{mode}'")
    return normalized


def get_2d_sincos_pos_embed_rect(
    embed_dim: int,
    grid_size: int | Sequence[int],
) -> np.ndarray:
    grid_h, grid_w = _to_2tuple(grid_size)
    grid_h_values = np.arange(grid_h, dtype=np.float32)
    grid_w_values = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w_values, grid_h_values)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_h, grid_w])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


class RectangularVisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim: int,
        pt_seq_len: int | Sequence[int],
        ft_seq_len: int | Sequence[int] | None = None,
        theta: float = 10000.0,
        num_prefix_tokens: int = 0,
    ):
        super().__init__()
        pt_grid_h, pt_grid_w = _to_2tuple(pt_seq_len)
        ft_grid_h, ft_grid_w = _to_2tuple(ft_seq_len if ft_seq_len is not None else pt_seq_len)

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim))

        t_h = torch.arange(ft_grid_h, dtype=torch.float32) / ft_grid_h * pt_grid_h
        t_w = torch.arange(ft_grid_w, dtype=torch.float32) / ft_grid_w * pt_grid_w

        freqs_h = torch.einsum("i,j->ij", t_h, freqs)
        freqs_w = torch.einsum("i,j->ij", t_w, freqs)
        freqs_h = torch.repeat_interleave(freqs_h, repeats=2, dim=-1)
        freqs_w = torch.repeat_interleave(freqs_w, repeats=2, dim=-1)
        freqs_h = freqs_h[:, None, :].expand(ft_grid_h, ft_grid_w, -1)
        freqs_w = freqs_w[None, :, :].expand(ft_grid_h, ft_grid_w, -1)
        freqs_2d = torch.cat([freqs_h, freqs_w], dim=-1)
        cos = freqs_2d.reshape(-1, freqs_2d.shape[-1]).cos()
        sin = freqs_2d.reshape(-1, freqs_2d.shape[-1]).sin()

        if num_prefix_tokens > 0:
            seq_dim = cos.shape[-1]
            cos_pad = torch.ones(num_prefix_tokens, seq_dim, dtype=cos.dtype)
            sin_pad = torch.zeros(num_prefix_tokens, seq_dim, dtype=sin.dtype)
            cos = torch.cat([cos_pad, cos], dim=0)
            sin = torch.cat([sin_pad, sin], dim=0)

        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.freqs_cos.shape[0]:
            raise ValueError(f"Expected sequence length {self.freqs_cos.shape[0]}, got {x.shape[-2]}")
        cos = self.freqs_cos.to(device=x.device, dtype=x.dtype)
        sin = self.freqs_sin.to(device=x.device, dtype=x.dtype)
        return x * cos + rotate_half(x) * sin


class ImageInputAdapter(nn.Module):
    def __init__(self, in_channels: int, target_channels: int = 3):
        super().__init__()
        self.in_channels = int(in_channels)
        self.target_channels = int(target_channels)
        self.proj = nn.Identity() if self.in_channels == self.target_channels else nn.Conv2d(
            self.in_channels,
            self.target_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    @property
    def is_identity(self) -> bool:
        return isinstance(self.proj, nn.Identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ImageOutputAdapter(nn.Module):
    def __init__(self, out_channels: int, source_channels: int = 3):
        super().__init__()
        self.out_channels = int(out_channels)
        self.source_channels = int(source_channels)
        self.proj = nn.Identity() if self.out_channels == self.source_channels else nn.Conv2d(
            self.source_channels,
            self.out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )

    @property
    def is_identity(self) -> bool:
        return isinstance(self.proj, nn.Identity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ConditionTypeStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=True),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} condition channels, got {condition.shape[1]}")
        return self.net(condition)


class ConditionStemCollection(nn.Module):
    def __init__(self, condition_channels_per_type: Sequence[int], cond_base_channels: int):
        super().__init__()
        if len(condition_channels_per_type) != 3:
            raise ValueError("RectangularConditionalJiT expects exactly 3 condition types")
        self.condition_channels_per_type = tuple(int(value) for value in condition_channels_per_type)
        self.cond_base_channels = int(cond_base_channels)
        self.stems = nn.ModuleList(
            [ConditionTypeStem(in_channels, self.cond_base_channels) for in_channels in self.condition_channels_per_type]
        )

    def forward(self, condition: torch.Tensor, condition_type_ids: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 4:
            raise ValueError(f"Expected condition to have shape [B, C, H, W], got {tuple(condition.shape)}")
        if condition_type_ids.ndim != 1 or condition_type_ids.shape[0] != condition.shape[0]:
            raise ValueError("condition_type_ids must be a 1D tensor with one entry per batch element")

        batch_size, channels, height, width = condition.shape
        output = condition.new_zeros((batch_size, self.cond_base_channels, height, width))

        for type_id, stem in enumerate(self.stems):
            sample_mask = condition_type_ids == type_id
            if not bool(sample_mask.any()):
                continue
            required_channels = self.condition_channels_per_type[type_id]
            if channels < required_channels:
                raise ValueError(
                    f"Condition tensor provides {channels} channels, but condition type {type_id} requires {required_channels}"
                )
            output[sample_mask] = stem(condition[sample_mask, :required_channels])
        return output


class ConditionTower(nn.Module):
    def __init__(
        self,
        input_size: int | Sequence[int],
        patch_size: int,
        in_channels: int,
        bottleneck_dim: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        g_token_count: int = 1,
        num_condition_types: int | None = None,
    ):
        super().__init__()
        self.input_size = _to_2tuple(input_size)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.g_token_count = int(g_token_count)
        self.num_condition_types = None if num_condition_types is None else int(num_condition_types)

        self.patch_embed = BottleneckPatchEmbed(
            self.input_size,
            self.patch_size,
            in_channels,
            bottleneck_dim,
            self.hidden_size,
            bias=True,
        )
        pos_embed = get_2d_sincos_pos_embed_rect(self.hidden_size, self.patch_embed.grid_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=True)
        self.g_token = nn.Parameter(torch.zeros(1, self.g_token_count, self.hidden_size))
        self.condition_type_embedding = (
            nn.Embedding(self.num_condition_types, self.hidden_size) if self.num_condition_types is not None else None
        )
        self.rope = RectangularVisionRotaryEmbeddingFast(
            dim=self.hidden_size // self.num_heads // 2,
            pt_seq_len=self.patch_embed.grid_size,
            num_prefix_tokens=self.g_token_count,
        )
        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(depth)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.g_token, std=0.02)
        if self.condition_type_embedding is not None:
            nn.init.normal_(self.condition_type_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.proj1.weight.view(self.patch_embed.proj1.weight.shape[0], -1))
        nn.init.xavier_uniform_(self.patch_embed.proj2.weight.view(self.patch_embed.proj2.weight.shape[0], -1))
        nn.init.constant_(self.patch_embed.proj2.bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        condition_features: torch.Tensor,
        timestep_embedding: torch.Tensor,
        condition_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_tokens = self.patch_embed(condition_features)
        spatial_tokens = spatial_tokens + self.pos_embed.to(device=spatial_tokens.device, dtype=spatial_tokens.dtype)
        global_tokens = self.g_token.expand(condition_features.shape[0], -1, -1).to(spatial_tokens.dtype)
        tokens = torch.cat([global_tokens, spatial_tokens], dim=1)
        if self.condition_type_embedding is not None and condition_type_ids is not None:
            if condition_type_ids.shape[0] != condition_features.shape[0]:
                raise ValueError("condition_type_ids must align with the condition batch size")
            type_embedding = self.condition_type_embedding(condition_type_ids).to(dtype=tokens.dtype)
            tokens = tokens + type_embedding.unsqueeze(1)
        for block in self.blocks:
            tokens = block(tokens, timestep_embedding, feat_rope=self.rope)
        return tokens[:, : self.g_token_count], tokens[:, self.g_token_count :], tokens


class GlobalConditionProjector(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size * expansion, hidden_size, bias=True),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.constant_(self.mlp[0].bias, 0)
        nn.init.normal_(self.mlp[2].weight, std=1e-3)
        nn.init.constant_(self.mlp[2].bias, 0)

    def forward(self, global_tokens: torch.Tensor) -> torch.Tensor:
        pooled = global_tokens.mean(dim=1)
        return self.mlp(self.norm(pooled))


class SparseCrossAttnAdapter(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.q_norm = RMSNorm(self.hidden_size)
        self.kv_norm = RMSNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.q_head_norm = RMSNorm(self.head_dim)
        self.k_head_norm = RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0)

    def forward(
        self,
        image_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        image_rope: RectangularVisionRotaryEmbeddingFast,
        condition_rope: RectangularVisionRotaryEmbeddingFast,
    ) -> torch.Tensor:
        batch_size, image_len, _ = image_tokens.shape
        cond_len = condition_tokens.shape[1]

        query = self.q_proj(self.q_norm(image_tokens)).reshape(batch_size, image_len, self.num_heads, self.head_dim)
        key = self.k_proj(self.kv_norm(condition_tokens)).reshape(batch_size, cond_len, self.num_heads, self.head_dim)
        value = self.v_proj(self.kv_norm(condition_tokens)).reshape(batch_size, cond_len, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        query = self.q_head_norm(query)
        key = self.k_head_norm(key)
        query = image_rope(query)
        key = condition_rope(key)

        attended = scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch_size, image_len, self.hidden_size)
        update = self.out_proj(attended)
        return image_tokens + self.alpha.to(update.dtype).view(1, 1, 1) * update


class FullJointMMDiTAdapter(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        self.image_norm1 = RMSNorm(self.hidden_size)
        self.cond_norm1 = RMSNorm(self.hidden_size)
        self.image_qkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.cond_qkv = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=True)
        self.image_q_head_norm = RMSNorm(self.head_dim)
        self.image_k_head_norm = RMSNorm(self.head_dim)
        self.cond_q_head_norm = RMSNorm(self.head_dim)
        self.cond_k_head_norm = RMSNorm(self.head_dim)
        self.image_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cond_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.image_norm2 = RMSNorm(self.hidden_size)
        self.cond_norm2 = RMSNorm(self.hidden_size)
        mlp_hidden_dim = int(self.hidden_size * mlp_ratio)
        self.image_mlp = SwiGLUFFN(self.hidden_size, mlp_hidden_dim)
        self.cond_mlp = SwiGLUFFN(self.hidden_size, mlp_hidden_dim)

        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear in [self.image_qkv, self.cond_qkv, self.image_out, self.cond_out]:
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0)

    def _project_qkv(self, tokens: torch.Tensor, proj: nn.Linear) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = tokens.shape
        qkv = proj(tokens).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(
        self,
        image_tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        image_rope: RectangularVisionRotaryEmbeddingFast,
        condition_rope: RectangularVisionRotaryEmbeddingFast,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_residual = image_tokens
        cond_residual = condition_tokens

        image_q, image_k, image_v = self._project_qkv(self.image_norm1(image_tokens), self.image_qkv)
        cond_q, cond_k, cond_v = self._project_qkv(self.cond_norm1(condition_tokens), self.cond_qkv)

        image_q = image_rope(self.image_q_head_norm(image_q))
        image_k = image_rope(self.image_k_head_norm(image_k))
        cond_q = condition_rope(self.cond_q_head_norm(cond_q))
        cond_k = condition_rope(self.cond_k_head_norm(cond_k))

        joint_q = torch.cat([cond_q, image_q], dim=2)
        joint_k = torch.cat([cond_k, image_k], dim=2)
        joint_v = torch.cat([cond_v, image_v], dim=2)

        joint_output = scaled_dot_product_attention(joint_q, joint_k, joint_v)
        cond_len = condition_tokens.shape[1]
        image_len = image_tokens.shape[1]
        cond_output, image_output = joint_output.split([cond_len, image_len], dim=2)
        cond_output = cond_output.transpose(1, 2).reshape(condition_tokens.shape[0], cond_len, self.hidden_size)
        image_output = image_output.transpose(1, 2).reshape(image_tokens.shape[0], image_len, self.hidden_size)

        condition_tokens = cond_residual + self.cond_out(cond_output)
        image_tokens = image_residual + self.alpha.to(image_output.dtype).view(1, 1, 1) * self.image_out(image_output)

        condition_tokens = condition_tokens + self.cond_mlp(self.cond_norm2(condition_tokens))
        image_tokens = image_tokens + self.beta.to(image_tokens.dtype).view(1, 1, 1) * self.image_mlp(self.image_norm2(image_tokens))
        return image_tokens, condition_tokens


@dataclass
class RectangularConditionalJiTModelOutput(BaseOutput):
    sample: torch.Tensor
    image_tokens: torch.Tensor | None = None
    condition_tokens: torch.Tensor | None = None
    global_tokens: torch.Tensor | None = None
    conditioning: torch.Tensor | None = None


class RectangularConditionalJiTModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        input_size: int | Sequence[int] = (512, 1024),
        patch_size: int = 32,
        image_in_channels: int = 3,
        image_out_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        condition_size: int | Sequence[int] | None = None,
        condition_channels_per_type: Sequence[int] = (35, 35, 35),
        cond_base_channels: int = 16,
        cond_bottleneck_dim: int | None = None,
        cond_tower_depth: int = 4,
        g_token_count: int = 1,
        interaction_mode: str = "sparse_xattn",
        interaction_layers: Sequence[int] | None = None,
        recompute_global_after_joint: bool = False,
        preset_name: str | None = "JiT-B/32",
    ):
        super().__init__()
        self.input_size = _to_2tuple(input_size)
        self.condition_size = _to_2tuple(condition_size if condition_size is not None else self.input_size)
        self.patch_size = int(patch_size)
        self.image_in_channels = int(image_in_channels)
        self.image_out_channels = int(image_out_channels)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.intermediate_out_channels = 3
        self.interaction_mode = _validate_interaction_mode(interaction_mode)
        resolved_interaction_layers = (
            tuple(int(layer) for layer in interaction_layers)
            if interaction_layers is not None
            else _default_interaction_layers(self.depth)
        )
        self.interaction_layers = tuple(sorted(dict.fromkeys(resolved_interaction_layers)))
        self.interaction_layer_set = set(self.interaction_layers)

        self.image_input_adapter = ImageInputAdapter(self.image_in_channels, target_channels=3)
        self.image_output_adapter = ImageOutputAdapter(self.image_out_channels, source_channels=self.intermediate_out_channels)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            self.input_size,
            self.patch_size,
            3,
            bottleneck_dim,
            self.hidden_size,
            bias=True,
        )
        image_pos_embed = get_2d_sincos_pos_embed_rect(self.hidden_size, self.x_embedder.grid_size)
        self.register_buffer("image_pos_embed", torch.from_numpy(image_pos_embed).float().unsqueeze(0), persistent=True)
        self.image_rope = RectangularVisionRotaryEmbeddingFast(
            dim=self.hidden_size // self.num_heads // 2,
            pt_seq_len=self.x_embedder.grid_size,
        )

        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(self.depth)
            ]
        )
        self.condition_stems = ConditionStemCollection(condition_channels_per_type, cond_base_channels)
        self.condition_tower = ConditionTower(
            input_size=self.condition_size,
            patch_size=self.patch_size,
            in_channels=cond_base_channels,
            bottleneck_dim=cond_bottleneck_dim if cond_bottleneck_dim is not None else bottleneck_dim,
            hidden_size=self.hidden_size,
            depth=cond_tower_depth,
            num_heads=self.num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            g_token_count=g_token_count,
            num_condition_types=len(condition_channels_per_type),
        )
        self.g_proj = GlobalConditionProjector(self.hidden_size)

        interaction_modules: dict[str, nn.Module] = {}
        for layer_index in self.interaction_layers:
            if self.interaction_mode == "sparse_xattn":
                interaction_modules[str(layer_index)] = SparseCrossAttnAdapter(self.hidden_size, self.num_heads)
            else:
                interaction_modules[str(layer_index)] = FullJointMMDiTAdapter(self.hidden_size, self.num_heads, mlp_ratio)
        self.interaction_blocks = nn.ModuleDict(interaction_modules)
        self.final_layer = FinalLayer(self.hidden_size, self.patch_size, self.intermediate_out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _init_linear_modules(root: nn.Module) -> None:
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        _init_linear_modules(self.blocks)
        _init_linear_modules(self.final_layer)
        nn.init.xavier_uniform_(self.x_embedder.proj1.weight.view(self.x_embedder.proj1.weight.shape[0], -1))
        nn.init.xavier_uniform_(self.x_embedder.proj2.weight.view(self.x_embedder.proj2.weight.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)
        nn.init.constant_(self.t_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.t_embedder.mlp[2].bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _prepare_timestep(self, timestep: torch.Tensor | float | int, batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=torch.float32)
        else:
            timestep = timestep.to(device=device, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep.view(1)
        if timestep.shape[0] == 1:
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} timesteps, got {timestep.shape[0]}")
        return timestep

    def _prepare_condition_type_ids(
        self,
        condition_type_ids: torch.Tensor | Sequence[int] | int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if torch.is_tensor(condition_type_ids):
            condition_type_ids = condition_type_ids.to(device=device, dtype=torch.long)
        elif isinstance(condition_type_ids, int):
            condition_type_ids = torch.tensor([condition_type_ids], device=device, dtype=torch.long)
        else:
            condition_type_ids = torch.tensor(list(condition_type_ids), device=device, dtype=torch.long)
        if condition_type_ids.ndim == 0:
            condition_type_ids = condition_type_ids.view(1)
        if condition_type_ids.shape[0] == 1:
            condition_type_ids = condition_type_ids.expand(batch_size)
        if condition_type_ids.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} condition types, got {condition_type_ids.shape[0]}")
        if int(condition_type_ids.min()) < 0 or int(condition_type_ids.max()) >= len(self.condition_stems.stems):
            raise ValueError("condition_type_ids contains an out-of-range value")
        return condition_type_ids

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        channels = self.intermediate_out_channels
        grid_h, grid_w = self.x_embedder.grid_size
        patch_h, patch_w = _to_2tuple(self.patch_size)
        if x.shape[1] != grid_h * grid_w:
            raise ValueError(f"Expected {grid_h * grid_w} tokens, got {x.shape[1]}")
        x = x.reshape(x.shape[0], grid_h, grid_w, patch_h, patch_w, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], channels, grid_h * patch_h, grid_w * patch_w)

    def encode_condition(
        self,
        condition: torch.Tensor,
        condition_type_ids: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_base = self.condition_stems(condition, condition_type_ids)
        global_tokens, condition_tokens, condition_sequence = self.condition_tower(
            condition_base,
            timestep_embedding,
            condition_type_ids=condition_type_ids,
        )
        conditioning = self._compute_global_conditioning(timestep_embedding, global_tokens)
        return global_tokens, condition_tokens, condition_sequence, conditioning

    def _compute_global_conditioning(
        self,
        timestep_embedding: torch.Tensor,
        global_tokens: torch.Tensor,
    ) -> torch.Tensor:
        global_embedding = self.g_proj(global_tokens)
        return timestep_embedding + global_embedding

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        condition: torch.Tensor,
        condition_type_ids: torch.Tensor | Sequence[int] | int,
        return_dict: bool = True,
        return_intermediates: bool = False,
    ) -> RectangularConditionalJiTModelOutput | tuple[torch.Tensor]:
        batch_size = sample.shape[0]
        timestep = self._prepare_timestep(timestep, batch_size, sample.device)
        condition_type_ids = self._prepare_condition_type_ids(condition_type_ids, batch_size, sample.device)
        timestep_embedding = self.t_embedder(timestep)

        global_tokens, condition_tokens, condition_sequence, conditioning = self.encode_condition(
            condition,
            condition_type_ids,
            timestep_embedding,
        )

        hidden_states = self.image_input_adapter(sample)
        hidden_states = self.x_embedder(hidden_states)
        hidden_states = hidden_states + self.image_pos_embed.to(device=hidden_states.device, dtype=hidden_states.dtype)

        interaction_condition_tokens = condition_sequence
        current_global_tokens = global_tokens
        for index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, conditioning, feat_rope=self.image_rope)
            if index not in self.interaction_layer_set:
                continue
            adapter = self.interaction_blocks[str(index)]
            if self.interaction_mode == "sparse_xattn":
                hidden_states = adapter(hidden_states, condition_sequence, self.image_rope, self.condition_tower.rope)
            else:
                hidden_states, interaction_condition_tokens = adapter(
                    hidden_states,
                    interaction_condition_tokens,
                    self.image_rope,
                    self.condition_tower.rope,
                )
                current_global_tokens = interaction_condition_tokens[:, : self.condition_tower.g_token_count]
                if self.config.recompute_global_after_joint:
                    conditioning = self._compute_global_conditioning(timestep_embedding, current_global_tokens)

        patch_predictions = self.final_layer(hidden_states, conditioning)
        sample = self.unpatchify(patch_predictions)
        sample = self.image_output_adapter(sample)

        if not return_dict:
            return (sample,)
        return RectangularConditionalJiTModelOutput(
            sample=sample,
            image_tokens=hidden_states if return_intermediates else None,
            condition_tokens=(interaction_condition_tokens if self.interaction_mode == "full_joint_mmdit" else condition_sequence)
            if return_intermediates
            else None,
            global_tokens=current_global_tokens if return_intermediates else None,
            conditioning=conditioning if return_intermediates else None,
        )

    def load_pretrained_jit_backbone(self, source_model: JiTModel) -> dict[str, Any]:
        if not isinstance(source_model, JiTModel):
            raise TypeError(f"Expected JiTModel, got {type(source_model)!r}")
        if source_model.config.patch_size != self.patch_size:
            raise ValueError(
                f"Patch size mismatch: source {source_model.config.patch_size}, target {self.patch_size}"
            )
        if source_model.config.hidden_size != self.hidden_size:
            raise ValueError(
                f"Hidden size mismatch: source {source_model.config.hidden_size}, target {self.hidden_size}"
            )
        if source_model.config.depth != self.depth:
            raise ValueError(f"Depth mismatch: source {source_model.config.depth}, target {self.depth}")
        if source_model.config.num_heads != self.num_heads:
            raise ValueError(
                f"Num heads mismatch: source {source_model.config.num_heads}, target {self.num_heads}"
            )

        self.t_embedder.load_state_dict(source_model.t_embedder.state_dict(), strict=True)
        self.x_embedder.load_state_dict(source_model.x_embedder.state_dict(), strict=True)
        for target_block, source_block in zip(self.blocks, source_model.blocks, strict=True):
            target_block.load_state_dict(source_block.state_dict(), strict=True)
        self.final_layer.load_state_dict(source_model.final_layer.state_dict(), strict=True)

        copied_pos_embed = False
        if hasattr(source_model, "pos_embed") and tuple(source_model.pos_embed.shape) == tuple(self.image_pos_embed.shape):
            self.image_pos_embed.copy_(source_model.pos_embed.detach())
            copied_pos_embed = True

        return {
            "copied_modules": ["t_embedder", "x_embedder", "blocks", "final_layer"],
            "copied_pos_embed": copied_pos_embed,
            "source_preset_name": getattr(source_model.config, "preset_name", None),
        }

    def load_pretrained_jit_backbone_from_preset(self, preset_name: str) -> dict[str, Any]:
        source_model = create_jit_model(preset_name)
        report = self.load_pretrained_jit_backbone(source_model)
        report["source"] = "preset"
        report["preset_name"] = preset_name
        return report

    def load_pretrained_jit_backbone_from_public_checkpoint(
        self,
        checkpoint_or_path: dict[str, Any] | str | Path,
        variant: str = "ema1",
        preset_name: str | None = None,
    ) -> dict[str, Any]:
        source_model, checkpoint, inferred = instantiate_model_from_public_checkpoint(
            checkpoint_or_path,
            preset_name=preset_name,
        )
        load_report = load_public_weights_into_model(source_model, checkpoint, variant=variant, strict=False)
        transplant_report = self.load_pretrained_jit_backbone(source_model)
        return {
            "source": "public_checkpoint",
            "variant": variant,
            "inferred": inferred,
            "load_report": load_report,
            "transplant_report": transplant_report,
        }


def create_rectangular_conditional_jit_model(
    preset_name: str = "JiT-B/32",
    **kwargs: Any,
) -> RectangularConditionalJiTModel:
    if preset_name not in JIT_PRESET_CONFIGS:
        available = ", ".join(sorted(JIT_PRESET_CONFIGS))
        raise ValueError(f"Unknown JiT preset '{preset_name}'. Available presets: {available}")
    base_config = dict(JIT_PRESET_CONFIGS[preset_name])
    base_config.pop("in_context_len", None)
    base_config.pop("in_context_start", None)
    base_config.update(kwargs)
    base_config.setdefault("preset_name", preset_name)
    base_config.setdefault("input_size", (512, 1024))
    base_config.setdefault("condition_size", base_config["input_size"])
    base_config.setdefault("image_in_channels", 3)
    base_config.setdefault("image_out_channels", 3)
    base_config.setdefault("interaction_mode", "sparse_xattn")
    base_config.setdefault("cond_tower_depth", 4 if base_config["patch_size"] == 32 else 6)
    base_config.setdefault("cond_base_channels", 16)
    base_config.setdefault("condition_channels_per_type", (35, 35, 35))
    return RectangularConditionalJiTModel(**base_config)