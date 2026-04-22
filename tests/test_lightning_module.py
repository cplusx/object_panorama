import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from models import RectangularConditionalJiTModel
from training.lightning_module import RectangularConditionalJiTLightningModule, _build_model_input_batch
from tools.train_lightning_rectangular_conditional_jit import _resolve_accumulate_grad_batches, _resolve_precision


class LightningModuleTests(unittest.TestCase):
    def _build_model_cfg(self) -> dict:
        return {
            "input_size": (64, 128),
            "patch_size": 16,
            "image_in_channels": 3,
            "image_out_channels": 3,
            "hidden_size": 64,
            "depth": 4,
            "num_heads": 4,
            "bottleneck_dim": 16,
            "condition_size": (64, 128),
            "condition_channels_per_type": (20,),
            "cond_base_channels": 8,
            "cond_bottleneck_dim": 12,
            "cond_tower_depth": 2,
            "interaction_mode": "sparse_xattn",
            "interaction_layers": (1, 3),
            "preset_name": None,
        }

    def _build_objective_cfg(self) -> dict:
        return {
            "name": "flow_matching",
            "t_min": 0.0,
            "t_max": 1.0,
            "noise_scale": 1.0,
            "use_model_rgb": False,
            "use_model_depth": True,
            "use_model_normal": True,
        }

    def _build_optim_cfg(self) -> dict:
        return {
            "lr": 1.0e-4,
            "backbone_lr_mult": 0.1,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "max_epochs": 1,
            "effective_steps_per_epoch": 10,
            "visualize_every_n_steps": 0,
            "lr_scheduler": {"name": "constant"},
        }

    def test_training_step_runs_on_fake_batch(self) -> None:
        torch.manual_seed(0)
        module = RectangularConditionalJiTLightningModule(
            model_cfg=self._build_model_cfg(),
            objective_cfg=self._build_objective_cfg(),
            loss_cfg={"mse_weight": 1.0, "l1_weight": 0.0},
            optim_cfg=self._build_optim_cfg(),
            freeze_cfg={},
            pretrained_cfg={},
        )

        batch = {
            "sample_ids": ["a", "b"],
            "model_rgb": torch.randn(2, 15, 64, 128),
            "model_depth": torch.randn(2, 5, 64, 128),
            "model_normal": torch.randn(2, 15, 64, 128),
            "edge_depth": torch.randn(2, 3, 64, 128),
            "meta": [{}, {}],
        }

        loss = module.training_step(batch, batch_idx=0)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_load_jit_false_skips_public_loader(self) -> None:
        with mock.patch.object(
            RectangularConditionalJiTModel,
            "load_pretrained_jit_backbone_from_public_checkpoint",
            autospec=True,
        ) as load_mock:
            RectangularConditionalJiTLightningModule(
                model_cfg=self._build_model_cfg(),
                objective_cfg=self._build_objective_cfg(),
                loss_cfg={"mse_weight": 1.0, "l1_weight": 0.0},
                optim_cfg=self._build_optim_cfg(),
                freeze_cfg={},
                pretrained_cfg={"load_jit": False, "public_checkpoint_path": "/tmp/mock.pt", "variant": "ema1"},
            )

        load_mock.assert_not_called()

    def test_load_jit_true_requires_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "pretrained.load_jit=true"):
            RectangularConditionalJiTLightningModule(
                model_cfg=self._build_model_cfg(),
                objective_cfg=self._build_objective_cfg(),
                loss_cfg={"mse_weight": 1.0, "l1_weight": 0.0},
                optim_cfg=self._build_optim_cfg(),
                freeze_cfg={},
                pretrained_cfg={"load_jit": True, "public_checkpoint_path": None, "variant": "ema1"},
            )

    def test_load_jit_true_calls_public_loader(self) -> None:
        with mock.patch.object(
            RectangularConditionalJiTModel,
            "load_pretrained_jit_backbone_from_public_checkpoint",
            autospec=True,
            return_value={"loaded": True},
        ) as load_mock:
            RectangularConditionalJiTLightningModule(
                model_cfg=self._build_model_cfg(),
                objective_cfg=self._build_objective_cfg(),
                loss_cfg={"mse_weight": 1.0, "l1_weight": 0.0},
                optim_cfg=self._build_optim_cfg(),
                freeze_cfg={},
                pretrained_cfg={"load_jit": True, "public_checkpoint_path": "/tmp/mock.pt", "variant": "ema1"},
            )

        load_mock.assert_called_once()

    def test_resolve_precision_only_allows_fp32(self) -> None:
        self.assertEqual(_resolve_precision(None, "32-true"), "32-true")
        self.assertEqual(_resolve_precision("fp32", "32-true"), "32-true")
        with self.assertRaises(ValueError):
            _resolve_precision("16-mixed", "32-true")
        with self.assertRaises(ValueError):
            _resolve_precision("bf16", "32-true")

    def test_resolve_accumulate_grad_batches_uses_effective_batch_size(self) -> None:
        self.assertEqual(_resolve_accumulate_grad_batches(batch_size=1, effective_batch_size=8, num_devices=2), 4)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            _resolve_accumulate_grad_batches(batch_size=2, effective_batch_size=7, num_devices=2)

    def test_training_visualization_logs_only_preview_to_wandb(self) -> None:
        module = RectangularConditionalJiTLightningModule(
            model_cfg=self._build_model_cfg(),
            objective_cfg=self._build_objective_cfg(),
            loss_cfg={"mse_weight": 1.0, "l1_weight": 0.0},
            optim_cfg=self._build_optim_cfg(),
            freeze_cfg={},
            pretrained_cfg={},
        )

        logged_payloads = []

        class FakeExperiment:
            def log(self, payload):
                logged_payloads.append(payload)

        fake_logger = SimpleNamespace(experiment=FakeExperiment())

        class FakeWandbImage:
            def __init__(self, path: str) -> None:
                self.path = path

        fake_wandb = SimpleNamespace(Image=FakeWandbImage)

        with tempfile.TemporaryDirectory() as tmp_dir:
            module._trainer = SimpleNamespace(default_root_dir=tmp_dir, global_step=7, is_global_zero=True, logger=fake_logger)
            debug_tensors = {
                "sample": torch.randn(1, 3, 64, 128),
                "condition": torch.randn(1, 20, 64, 128),
                "pred": torch.randn(1, 3, 64, 128),
                "target": torch.randn(1, 3, 64, 128),
            }

            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                module._save_training_visualization(debug_tensors)

            preview_path = Path(tmp_dir) / "visuals" / "step_000008" / "preview.png"
            self.assertTrue(preview_path.is_file())
            with Image.open(preview_path) as image:
                self.assertEqual(image.size, (128 * 4, 64))

        self.assertEqual(len(logged_payloads), 1)
        self.assertEqual(logged_payloads[0]["global_step"], 7)
        self.assertIn("train/preview", logged_payloads[0])
        self.assertNotIn("condition", logged_payloads[0])

    def test_model_input_batch_can_disable_condition_dropout(self) -> None:
        batch = {
            "sample_ids": ["a", "b"],
            "model_rgb": torch.ones(2, 15, 64, 128),
            "model_depth": torch.ones(2, 5, 64, 128),
            "model_normal": torch.ones(2, 15, 64, 128),
            "edge_depth": torch.ones(2, 3, 64, 128),
            "meta": [{}, {}],
        }
        objective_cfg = self._build_objective_cfg()
        objective_cfg["condition_dropout_p"] = 1.0

        train_model_input = _build_model_input_batch(batch, objective_cfg, enable_condition_dropout=True)
        val_model_input = _build_model_input_batch(batch, objective_cfg, enable_condition_dropout=False)

        self.assertTrue(torch.equal(train_model_input.condition, torch.zeros_like(train_model_input.condition)))
        self.assertFalse(torch.equal(val_model_input.condition, torch.zeros_like(val_model_input.condition)))


if __name__ == "__main__":
    unittest.main()