import unittest

import torch

from training import build_edge3d_condition
from training.lightning_module import RectangularConditionalJiTLightningModule
from tools.overfit_rectangular_conditional_jit import _prepare_model_cfg as prepare_overfit_model_cfg
from tools.train_rectangular_conditional_jit import _prepare_model_cfg as prepare_train_model_cfg


def _make_fake_batch() -> dict[str, torch.Tensor | list]:
    return {
        "sample_ids": ["a", "b"],
        "model_rgb": torch.randn(2, 15, 64, 128),
        "model_depth": torch.randn(2, 5, 64, 128),
        "model_normal": torch.randn(2, 15, 64, 128),
        "edge_depth": torch.randn(2, 3, 64, 128),
        "meta": [{}, {}],
    }


class Edge3DConditionChannelTests(unittest.TestCase):
    def test_default_condition_channels_is_20(self) -> None:
        torch.manual_seed(0)
        batch = _make_fake_batch()
        condition = build_edge3d_condition(batch)
        self.assertEqual(tuple(condition.shape), (2, 20, 64, 128))

    def test_lightning_module_raises_if_model_still_expects_35_channels(self) -> None:
        torch.manual_seed(0)
        module = RectangularConditionalJiTLightningModule(
            model_cfg={
                "input_size": (64, 128),
                "patch_size": 16,
                "image_in_channels": 3,
                "image_out_channels": 3,
                "hidden_size": 64,
                "depth": 4,
                "num_heads": 4,
                "bottleneck_dim": 16,
                "condition_size": (64, 128),
                "condition_channels_per_type": (35,),
                "cond_base_channels": 8,
                "cond_bottleneck_dim": 12,
                "cond_tower_depth": 2,
                "interaction_mode": "sparse_xattn",
                "interaction_layers": (1, 3),
                "preset_name": None,
            },
            objective_cfg={
                "name": "flow_matching",
                "t_min": 0.0,
                "t_max": 1.0,
                "noise_scale": 1.0,
                "use_model_rgb": False,
                "use_model_depth": True,
                "use_model_normal": True,
            },
            loss_cfg={"mse_weight": 1.0, "l1_weight": 0.0},
            optim_cfg={
                "lr": 1.0e-4,
                "backbone_lr_mult": 0.1,
                "weight_decay": 0.0,
                "optimizer": "adamw",
                "max_steps": 10,
                "lr_scheduler": {"name": "constant"},
            },
            freeze_cfg={},
            pretrained_cfg={},
        )

        with self.assertRaisesRegex(
            ValueError,
            "Condition channel mismatch: model expects 35, but objective built 20",
        ):
            module.training_step(_make_fake_batch(), batch_idx=0)

    def test_simple_train_entrypoints_expand_single_condition_channel_slot(self) -> None:
        model_cfg = {"name": "rectangular_conditional_jit", "condition_channels_per_type": [20]}
        self.assertEqual(prepare_overfit_model_cfg(model_cfg)["condition_channels_per_type"], [20, 20, 20])
        self.assertEqual(prepare_train_model_cfg(model_cfg)["condition_channels_per_type"], [20, 20, 20])


if __name__ == "__main__":
    unittest.main()