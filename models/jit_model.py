from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from torch import nn

from .jit_layers import (
    BottleneckPatchEmbed,
    FinalLayer,
    JiTBlock,
    LabelEmbedder,
    TimestepEmbedder,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)


JIT_PRESET_CONFIGS = {
    "JiT-B/16": {
        "depth": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 4,
        "patch_size": 16,
    },
    "JiT-B/32": {
        "depth": 12,
        "hidden_size": 768,
        "num_heads": 12,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 4,
        "patch_size": 32,
    },
    "JiT-L/16": {
        "depth": 24,
        "hidden_size": 1024,
        "num_heads": 16,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 8,
        "patch_size": 16,
    },
    "JiT-L/32": {
        "depth": 24,
        "hidden_size": 1024,
        "num_heads": 16,
        "bottleneck_dim": 128,
        "in_context_len": 32,
        "in_context_start": 8,
        "patch_size": 32,
    },
    "JiT-H/16": {
        "depth": 32,
        "hidden_size": 1280,
        "num_heads": 16,
        "bottleneck_dim": 256,
        "in_context_len": 32,
        "in_context_start": 10,
        "patch_size": 16,
    },
    "JiT-H/32": {
        "depth": 32,
        "hidden_size": 1280,
        "num_heads": 16,
        "bottleneck_dim": 256,
        "in_context_len": 32,
        "in_context_start": 10,
        "patch_size": 32,
    },
}


@dataclass
class JiTModelOutput(BaseOutput):
    sample: torch.Tensor


class JiTModel(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        input_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_classes: int = 1000,
        bottleneck_dim: int = 128,
        in_context_len: int = 32,
        in_context_start: int = 8,
        preset_name: str | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            input_size,
            patch_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
            bias=True,
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            nn.init.normal_(self.in_context_posemb, std=0.02)
        else:
            self.register_parameter("in_context_posemb", None)

        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len, num_cls_token=0)
        self.feat_rope_incontext = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len,
        )

        self.blocks = nn.ModuleList(
            [
                JiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop if (depth // 4 * 3 > index >= depth // 4) else 0.0,
                    proj_drop=proj_drop if (depth // 4 * 3 > index >= depth // 4) else 0.0,
                )
                for index in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        weight1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(weight1.view([weight1.shape[0], -1]))
        weight2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(weight2.view([weight2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        channels = self.out_channels
        grid_height = grid_width = int(x.shape[1] ** 0.5)
        if grid_height * grid_width != x.shape[1]:
            raise ValueError(f"Expected a square token grid, got {x.shape[1]} tokens")
        x = x.reshape(x.shape[0], grid_height, grid_width, patch_size, patch_size, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], channels, grid_height * patch_size, grid_width * patch_size)

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

    def _prepare_class_labels(
        self,
        class_labels: torch.Tensor | list[int] | tuple[int, ...],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if class_labels is None:
            raise ValueError("JiTModel requires class_labels")
        if not torch.is_tensor(class_labels):
            class_labels = torch.tensor(class_labels, device=device, dtype=torch.long)
        else:
            class_labels = class_labels.to(device=device, dtype=torch.long)
        if class_labels.ndim == 0:
            class_labels = class_labels.view(1)
        if class_labels.shape[0] == 1:
            class_labels = class_labels.expand(batch_size)
        if class_labels.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} class labels, got {class_labels.shape[0]}")
        return class_labels

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        class_labels: torch.Tensor | list[int] | tuple[int, ...] | None,
        return_dict: bool = True,
    ) -> JiTModelOutput | tuple[torch.Tensor]:
        batch_size = sample.shape[0]
        timestep = self._prepare_timestep(timestep, batch_size, sample.device)
        class_labels = self._prepare_class_labels(class_labels, batch_size, sample.device)

        timestep_embedding = self.t_embedder(timestep)
        label_embedding = self.y_embedder(class_labels)
        conditioning = timestep_embedding + label_embedding

        hidden_states = self.x_embedder(sample)
        hidden_states = hidden_states + self.pos_embed.to(hidden_states.dtype)

        for index, block in enumerate(self.blocks):
            if self.in_context_len > 0 and index == self.in_context_start:
                in_context_tokens = label_embedding.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens = in_context_tokens + self.in_context_posemb.to(in_context_tokens.dtype)
                hidden_states = torch.cat([in_context_tokens, hidden_states], dim=1)
            rope = self.feat_rope if index < self.in_context_start else self.feat_rope_incontext
            hidden_states = block(hidden_states, conditioning, feat_rope=rope)

        hidden_states = hidden_states[:, self.in_context_len :]
        hidden_states = self.final_layer(hidden_states, conditioning)
        sample = self.unpatchify(hidden_states, self.patch_size)

        if not return_dict:
            return (sample,)
        return JiTModelOutput(sample=sample)


def create_jit_model(preset_name: str, **kwargs) -> JiTModel:
    if preset_name not in JIT_PRESET_CONFIGS:
        available = ", ".join(sorted(JIT_PRESET_CONFIGS))
        raise ValueError(f"Unknown JiT preset '{preset_name}'. Available presets: {available}")
    config = dict(JIT_PRESET_CONFIGS[preset_name])
    config.update(kwargs)
    config.setdefault("preset_name", preset_name)
    return JiTModel(**config)