from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..jit_layers import get_2d_sincos_pos_embed_from_grid, rotate_half


def _to_2tuple(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple, got {value!r}")
        return int(value[0]), int(value[1])
    size = int(value)
    return size, size


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