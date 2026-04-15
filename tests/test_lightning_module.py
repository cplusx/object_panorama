import unittest

import torch

from training.lightning_module import RectangularConditionalJiTLightningModule


class LightningModuleTests(unittest.TestCase):
    def test_training_step_runs_on_fake_batch(self) -> None:
        torch.manual_seed(0)
        module = RectangularConditionalJiTLightningModule(
            model_cfg={
                "input_size": (64, 128),
                "patch_size": 16,
                "image_in_channels": 1,
                "image_out_channels": 1,
                "hidden_size": 64,
                "depth": 4,
                "num_heads": 4,
                "bottleneck_dim": 16,
                "condition_size": (64, 128),
                "condition_channels_per_type": (5, 5, 5),
                "cond_base_channels": 8,
                "cond_bottleneck_dim": 12,
                "cond_tower_depth": 2,
                "interaction_mode": "sparse_xattn",
                "interaction_layers": (1, 3),
                "preset_name": None,
            },
            objective_cfg={"name": "paired_supervised", "fixed_timestep": 0.5},
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

        batch = {
            "sample_ids": ["a", "b"],
            "input": torch.randn(2, 1, 64, 128),
            "condition": torch.randn(2, 5, 64, 128),
            "target": torch.randn(2, 1, 64, 128),
            "condition_type_ids": torch.tensor([0, 2], dtype=torch.long),
            "meta": [{}, {}],
        }

        loss = module.training_step(batch, batch_idx=0)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()