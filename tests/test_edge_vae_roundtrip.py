from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import torch

from edge3d.tensor_format import save_mixed_precision_sample
from edge_vae.roundtrip import run_edge_vae_roundtrip


class _IdentityCodec:
    def __init__(self, *_args, **_kwargs):
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32

    def roundtrip(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _channels, height, width = x.shape
        latents = torch.zeros((batch, 4, max(1, height // 8), max(1, width // 8)), dtype=torch.float32)
        return {
            "input": x.to(dtype=torch.float32),
            "latents": latents,
            "decoded": x.to(dtype=torch.float32),
        }


class TestEdgeVaeRoundtrip(unittest.TestCase):
    @mock.patch("edge_vae.roundtrip.DiffusersVAECodec", _IdentityCodec)
    def test_raw_roundtrip_saves_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = self._create_sample(Path(temp_dir) / "sample_raw.npz")
            output_dir = Path(temp_dir) / "raw_outputs"

            result = run_edge_vae_roundtrip(
                sample_path=str(sample_path),
                output_dir=str(output_dir),
                mode="raw",
                vae_cfg={"pretrained_model_name_or_path": "fake", "torch_dtype": "float32"},
                transform_cfg={"depth_scale": 2.0, "raw_scale": 1.0 / np.sqrt(3.0), "decode_valid_threshold": 0.02},
                runtime_cfg={"device": "cpu"},
            )

            self.assertEqual(result["mode"], "raw")
            self.assertTrue((output_dir / "input_raw_encoded.pt").is_file())
            self.assertTrue((output_dir / "decoded_raw_encoded.pt").is_file())
            self.assertTrue((output_dir / "decoded_edge_depth.pt").is_file())
            self.assertTrue((output_dir / "preview.png").is_file())
            self.assertTrue((output_dir / "target_edge_points.ply").is_file())
            self.assertTrue((output_dir / "pred_edge_points.ply").is_file())
            self.assertTrue((output_dir / "overlap_pointcloud.ply").is_file())

    @mock.patch("edge_vae.roundtrip.DiffusersVAECodec", _IdentityCodec)
    def test_df_roundtrip_saves_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_path = self._create_sample(Path(temp_dir) / "sample_df.npz")
            output_dir = Path(temp_dir) / "df_outputs"

            result = run_edge_vae_roundtrip(
                sample_path=str(sample_path),
                output_dir=str(output_dir),
                mode="df",
                vae_cfg={"pretrained_model_name_or_path": "fake", "torch_dtype": "float32"},
                transform_cfg={"depth_scale": 2.0, "beta": 30.0, "decode_valid_threshold": 0.02},
                runtime_cfg={"device": "cpu"},
            )

            self.assertEqual(result["mode"], "df")
            self.assertTrue((output_dir / "input_df_encoded.pt").is_file())
            self.assertTrue((output_dir / "decoded_df_encoded.pt").is_file())
            self.assertTrue((output_dir / "decoded_edge_depth.pt").is_file())
            self.assertTrue((output_dir / "preview.png").is_file())
            self.assertTrue((output_dir / "target_edge_points.ply").is_file())
            self.assertTrue((output_dir / "pred_edge_points.ply").is_file())
            self.assertTrue((output_dir / "overlap_pointcloud.ply").is_file())

    def _create_sample(self, sample_path: Path) -> Path:
        height, width = 8, 16
        model_tensor = np.zeros((35, height, width), dtype=np.float32)
        edge_tensor = np.zeros((3, height, width), dtype=np.float32)
        edge_tensor[0, 2:6, 4] = 1.0
        edge_tensor[1, 1:7, 7] = 1.5
        edge_tensor[2, 3:5, 10] = 0.8
        save_mixed_precision_sample(
            sample_path,
            uid=sample_path.stem,
            model_tensor=model_tensor,
            edge_tensor=edge_tensor,
            resolution=height,
            model_max_hits=5,
            edge_max_hits=3,
        )
        return sample_path


if __name__ == "__main__":
    unittest.main()