import unittest

import torch

from models import RectangularConditionalJiTModel
from training import run_train_step


def _assert_nonzero_grad(test_case: unittest.TestCase, tensor: torch.Tensor | None, name: str) -> None:
    test_case.assertIsNotNone(tensor, msg=f"{name} grad is missing")
    test_case.assertGreater(float(tensor.abs().sum().item()), 0.0, msg=f"{name} grad is zero")


class TrainStepTests(unittest.TestCase):
    def _make_model(self) -> RectangularConditionalJiTModel:
        return RectangularConditionalJiTModel(
            input_size=(64, 128),
            patch_size=16,
            image_in_channels=3,
            image_out_channels=3,
            hidden_size=64,
            depth=4,
            num_heads=4,
            bottleneck_dim=16,
            condition_size=(64, 128),
            condition_channels_per_type=(20, 20, 20),
            cond_base_channels=8,
            cond_bottleneck_dim=12,
            cond_tower_depth=2,
            interaction_mode="sparse_xattn",
            interaction_layers=(1, 3),
            preset_name=None,
        )

    def _activate_conditioning_path(self, model: RectangularConditionalJiTModel) -> None:
        with torch.no_grad():
            for block in model.blocks:
                block.adaLN_modulation[-1].weight.normal_(mean=0.0, std=0.02)
                block.adaLN_modulation[-1].bias.zero_()
            model.final_layer.adaLN_modulation[-1].weight.normal_(mean=0.0, std=0.02)
            model.final_layer.adaLN_modulation[-1].bias.zero_()
            model.final_layer.linear.weight.normal_(mean=0.0, std=0.02)
            model.final_layer.linear.bias.zero_()
            model.interaction_blocks["1"].alpha.fill_(1.0)

    def test_run_train_step_and_backward(self) -> None:
        torch.manual_seed(0)
        model = self._make_model()
        self._activate_conditioning_path(model)

        batch = {
            "sample_ids": ["a", "b"],
            "model_rgb": torch.randn(2, 15, 64, 128),
            "model_depth": torch.randn(2, 5, 64, 128),
            "model_normal": torch.randn(2, 15, 64, 128),
            "edge_depth": torch.randn(2, 3, 64, 128),
            "meta": [{}, {}],
        }
        output = run_train_step(
            model,
            batch,
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
            device="cpu",
        )
        self.assertIn("loss_mse", output.loss_dict)
        self.assertIsNotNone(output.loss_total)
        self.assertEqual(tuple(output.target.shape), (2, 3, 64, 128))
        self.assertEqual(tuple(output.sample.shape), (2, 3, 64, 128))
        self.assertEqual(tuple(output.condition.shape), (2, 20, 64, 128))

        output.loss_total.backward()
        _assert_nonzero_grad(self, model.g_proj.mlp[2].weight.grad, "g_proj")
        _assert_nonzero_grad(self, model.condition_stems.stems[0].net[0].weight.grad, "condition stem")
        _assert_nonzero_grad(self, model.interaction_blocks["1"].q_proj.weight.grad, "interaction block")


if __name__ == "__main__":
    unittest.main()