import torch
from torch import nn

from ..jit_layers import RMSNorm, SwiGLUFFN, scaled_dot_product_attention
from .geometry import RectangularVisionRotaryEmbeddingFast


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