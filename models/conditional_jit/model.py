from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from torch import nn

from ..jit_checkpoint_loader import instantiate_model_from_public_checkpoint, load_public_weights_into_model
from ..jit_layers import BottleneckPatchEmbed, FinalLayer, JiTBlock, TimestepEmbedder
from ..jit_model import JIT_PRESET_CONFIGS, JiTModel, create_jit_model
from .adapters import (
    FullJointMMDiTAdapter,
    GlobalConditionProjector,
    ImageInputAdapter,
    ImageOutputAdapter,
    SparseCrossAttnAdapter,
)
from .condition import ConditionStemCollection, ConditionTower, ConditionTypeStem
from .geometry import RectangularVisionRotaryEmbeddingFast, _to_2tuple, get_2d_sincos_pos_embed_rect


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

    def _compute_global_conditioning(
        self,
        timestep_embedding: torch.Tensor,
        global_tokens: torch.Tensor,
    ) -> torch.Tensor:
        global_embedding = self.g_proj(global_tokens)
        return timestep_embedding + global_embedding

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