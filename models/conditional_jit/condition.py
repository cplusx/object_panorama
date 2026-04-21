from typing import Sequence

import torch
from torch import nn

from ..jit_layers import BottleneckPatchEmbed, JiTBlock
from .geometry import RectangularVisionRotaryEmbeddingFast, _to_2tuple, get_2d_sincos_pos_embed_rect


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
        output = None

        for type_id, stem in enumerate(self.stems):
            sample_mask = condition_type_ids == type_id
            if not bool(sample_mask.any()):
                continue
            required_channels = self.condition_channels_per_type[type_id]
            if channels < required_channels:
                raise ValueError(
                    f"Condition tensor provides {channels} channels, but condition type {type_id} requires {required_channels}"
                )
            stem_output = stem(condition[sample_mask, :required_channels])
            if output is None:
                output = stem_output.new_zeros((batch_size, self.cond_base_channels, height, width))
            elif output.dtype != stem_output.dtype:
                output = output.to(dtype=stem_output.dtype)
            output[sample_mask] = stem_output
        if output is None:
            output = condition.new_zeros((batch_size, self.cond_base_channels, height, width))
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