import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.run_edge3d_x0_bridge_inference import (
    _load_checkpoint_payload,
    resolve_checkpoint_path,
    resolve_inference_settings,
    save_inference_pointclouds,
)


class InferenceScriptTests(unittest.TestCase):
    def test_resolve_inference_settings_prefers_inference_section(self) -> None:
        config = {
            "validation": {"num_inference_steps": 20, "cfg_scale": 1.5, "inference_dtype": "float16"},
            "inference": {"num_inference_steps": 64, "cfg_scale": 2.25},
        }

        resolved = resolve_inference_settings(config)

        self.assertEqual(resolved["num_steps"], 64)
        self.assertEqual(resolved["cfg_scale"], 2.25)
        self.assertEqual(resolved["inference_dtype"], "float16")

    def test_resolve_inference_settings_allows_explicit_overrides(self) -> None:
        config = {
            "validation": {"num_inference_steps": 20, "cfg_scale": 1.5, "inference_dtype": "float16"},
            "inference": {"num_inference_steps": 64, "cfg_scale": 2.25, "inference_dtype": "float32"},
        }

        resolved = resolve_inference_settings(config, num_steps=12, cfg_scale=3.0, inference_dtype="bfloat16")

        self.assertEqual(resolved["num_steps"], 12)
        self.assertEqual(resolved["cfg_scale"], 3.0)
        self.assertEqual(resolved["inference_dtype"], "bfloat16")

    def test_resolve_checkpoint_path_auto_uses_last_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {"output_dir": tmp_dir, "experiment_name": "edge3d_flow_train"}
            checkpoint_dir = Path(tmp_dir) / "edge3d_flow_train" / "checkpoints" / "last.ckpt"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            resolved = resolve_checkpoint_path("auto", config)

        self.assertEqual(resolved, checkpoint_dir.resolve())

    def test_load_checkpoint_payload_uses_deepspeed_zero_checkpoint_for_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "last.ckpt"
            (checkpoint_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
            with mock.patch(
                "deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint",
                return_value={"model.weight": np.array([1.0], dtype=np.float32)},
            ) as loader:
                payload = _load_checkpoint_payload(checkpoint_dir)

        self.assertIn("model.weight", payload)
        loader.assert_called_once_with(str(checkpoint_dir))

    def test_save_inference_pointclouds_writes_expected_plys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            height = 4
            width = 8
            model_depth = np.full((1, height, width), 1.0, dtype=np.float32)
            pred_edge_depth = np.full((1, height, width), 1.25, dtype=np.float32)
            target_edge_depth = np.full((1, height, width), 1.5, dtype=np.float32)

            outputs = save_inference_pointclouds(
                tmp_dir,
                model_depth=model_depth,
                pred_edge_depth=pred_edge_depth,
                target_edge_depth=target_edge_depth,
            )

            self.assertEqual(
                set(outputs.keys()),
                {
                    "model_points",
                    "pred_edge_points",
                    "target_edge_points",
                    "target_pred_points",
                    "model_target_pred_points",
                },
            )
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), path)


if __name__ == "__main__":
    unittest.main()