from __future__ import annotations

import sys
from typing import Any

import torch


class DiffusersVAECodec:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype: str = "float16",
        device: str = "cuda",
        subfolder: str | None = None,
        from_pretrained_kwargs: dict[str, Any] | None = None,
    ):
        self.device = _resolve_device(device)
        self.torch_dtype = _resolve_torch_dtype(torch_dtype, self.device)
        autoencoder_kl_cls = _load_autoencoder_kl()
        load_kwargs: dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
        }
        if subfolder is not None:
            load_kwargs["subfolder"] = str(subfolder)
        if from_pretrained_kwargs is not None:
            load_kwargs.update(dict(from_pretrained_kwargs))
        self.vae = autoencoder_kl_cls.from_pretrained(
            pretrained_model_name_or_path,
            **load_kwargs,
        )
        self.vae.to(device=self.device, dtype=self.torch_dtype)
        self.vae.eval()
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 1.0))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        prepared = _prepare_image_tensor(x, device=self.device, dtype=self.torch_dtype)
        posterior = self.vae.encode(prepared).latent_dist
        return posterior.mode() * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        prepared = _prepare_latent_tensor(z, device=self.device, dtype=self.torch_dtype)
        return self.vae.decode(prepared / self.scaling_factor).sample

    @torch.no_grad()
    def roundtrip(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        prepared = _prepare_image_tensor(x, device=self.device, dtype=self.torch_dtype)
        latents = self.encode(prepared)
        decoded = self.decode(latents)
        return {
            "input": prepared,
            "latents": latents,
            "decoded": decoded,
        }


def _load_autoencoder_kl():
    try:
        from diffusers import AutoencoderKL
        return AutoencoderKL
    except Exception as exc:
        if not _is_known_diffusers_custom_op_error(exc):
            raise RuntimeError("Failed to import diffusers AutoencoderKL for edge_vae codec") from exc

    original_custom_op = torch.library.custom_op
    original_register_fake = torch.library.register_fake
    try:
        torch.library.custom_op = _no_op_decorator
        torch.library.register_fake = _no_op_decorator
        _purge_diffusers_modules()
        from diffusers import AutoencoderKL
    except Exception as exc:
        raise RuntimeError("Failed to import diffusers AutoencoderKL for edge_vae codec") from exc
    finally:
        torch.library.custom_op = original_custom_op
        torch.library.register_fake = original_register_fake
    return AutoencoderKL


def _is_known_diffusers_custom_op_error(exc: Exception) -> bool:
    message = str(exc)
    return "infer_schema(func)" in message and "unsupported type torch.Tensor" in message


def _no_op_decorator(*args, **kwargs):
    del kwargs

    def wrap(func):
        return func

    if args and callable(args[0]) and len(args) == 1:
        return args[0]
    return wrap


def _purge_diffusers_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "diffusers" or module_name.startswith("diffusers."):
            sys.modules.pop(module_name, None)


def _resolve_device(device: str) -> torch.device:
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    normalized = str(dtype_name).strip().lower()
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'")
    resolved = mapping[normalized]
    if device.type == "cpu" and resolved == torch.float16:
        return torch.float32
    return resolved


def _prepare_image_tensor(x: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor input, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"Expected image tensor with shape [B, C, H, W], got {tuple(x.shape)}")
    if x.shape[1] != 3:
        raise ValueError(f"Expected 3-channel tensor for SDXL VAE, got {tuple(x.shape)}")
    return x.to(device=device, dtype=dtype).clamp_(-1.0, 1.0)


def _prepare_latent_tensor(z: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(z):
        raise TypeError(f"Expected torch.Tensor input, got {type(z)!r}")
    if z.ndim != 4:
        raise ValueError(f"Expected latent tensor with shape [B, C, H, W], got {tuple(z.shape)}")
    return z.to(device=device, dtype=dtype)