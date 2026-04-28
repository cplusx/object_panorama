from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from edge_vae.codec import DiffusersVAECodec


class _FakeLatentDistribution:
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def mode(self) -> torch.Tensor:
        return self._tensor + 1.0


class _FakeAutoencoderKL:
    last_from_pretrained = None

    def __init__(self, *, torch_dtype: torch.dtype):
        self.config = SimpleNamespace(scaling_factor=0.5)
        self.loaded_torch_dtype = torch_dtype
        self.device = None
        self.dtype = torch_dtype

    @classmethod
    def from_pretrained(cls, name: str, **kwargs):
        cls.last_from_pretrained = {
            "name": name,
            **kwargs,
        }
        torch_dtype = kwargs["torch_dtype"]
        return cls(torch_dtype=torch_dtype)

    def to(self, device=None, dtype=None):
        self.device = torch.device(device)
        self.dtype = dtype if dtype is not None else self.dtype
        return self

    def eval(self):
        return self

    def encode(self, tensor: torch.Tensor):
        self.last_encode_input = tensor.detach().clone()
        return SimpleNamespace(latent_dist=_FakeLatentDistribution(tensor))

    def decode(self, latents: torch.Tensor):
        self.last_decode_input = latents.detach().clone()
        return SimpleNamespace(sample=latents - 1.0)


class TestEdgeVaeCodec(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAutoencoderKL.last_from_pretrained = None

    @mock.patch("edge_vae.codec._load_autoencoder_kl", return_value=_FakeAutoencoderKL)
    def test_roundtrip_uses_diffusers_style_encode_decode_scaling(self, _load_autoencoder_kl_mock) -> None:
        codec = DiffusersVAECodec(device="cpu", torch_dtype="float32")
        image = torch.zeros((1, 3, 8, 8), dtype=torch.float32)

        roundtrip = codec.roundtrip(image)

        self.assertEqual(tuple(roundtrip["input"].shape), (1, 3, 8, 8))
        self.assertTrue(torch.allclose(roundtrip["decoded"], image))
        self.assertTrue(torch.allclose(roundtrip["latents"], torch.full_like(image, 0.5)))

    @mock.patch("edge_vae.codec._load_autoencoder_kl", return_value=_FakeAutoencoderKL)
    def test_cpu_falls_back_to_float32(self, _load_autoencoder_kl_mock) -> None:
        codec = DiffusersVAECodec(device="cpu", torch_dtype="float16")
        self.assertEqual(codec.torch_dtype, torch.float32)
        self.assertEqual(codec.vae.loaded_torch_dtype, torch.float32)

    @mock.patch("edge_vae.codec._load_autoencoder_kl", return_value=_FakeAutoencoderKL)
    def test_forwards_optional_subfolder_to_from_pretrained(self, _load_autoencoder_kl_mock) -> None:
        DiffusersVAECodec(
            pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell",
            subfolder="vae",
            device="cpu",
            torch_dtype="float32",
        )
        self.assertEqual(
            _FakeAutoencoderKL.last_from_pretrained,
            {
                "name": "black-forest-labs/FLUX.1-schnell",
                "subfolder": "vae",
                "torch_dtype": torch.float32,
            },
        )


if __name__ == "__main__":
    unittest.main()