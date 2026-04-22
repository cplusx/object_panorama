import unittest

import torch

from training import build_edge3d_condition, build_jit_flow_matching_batch


class TrainingObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.batch = {
            "model_rgb": torch.randn(2, 15, 8, 16),
            "model_depth": torch.randn(2, 5, 8, 16),
            "model_normal": torch.randn(2, 15, 8, 16),
            "edge_depth": torch.randn(2, 3, 8, 16),
        }

    def test_build_edge3d_condition_defaults_to_depth_and_normal(self) -> None:
        condition = build_edge3d_condition(self.batch)
        self.assertEqual(tuple(condition.shape), (2, 20, 8, 16))

    def test_build_jit_flow_matching_batch(self) -> None:
        model_input = build_jit_flow_matching_batch(
            self.batch,
            t_min=0.0,
            t_max=1.0,
            noise_scale=1.0,
            condition_type_id=0,
        )
        self.assertEqual(tuple(model_input.sample.shape), (2, 3, 8, 16))
        self.assertEqual(tuple(model_input.condition.shape), (2, 20, 8, 16))
        self.assertTrue(torch.equal(model_input.target, self.batch["edge_depth"]))
        self.assertEqual(tuple(model_input.condition_type_ids.shape), (2,))
        self.assertFalse(torch.allclose(model_input.sample, model_input.target))

    def test_build_jit_flow_matching_batch_zeroes_condition_when_dropout_is_one(self) -> None:
        model_input = build_jit_flow_matching_batch(
            self.batch,
            t_min=0.0,
            t_max=1.0,
            noise_scale=1.0,
            condition_dropout_p=1.0,
            condition_type_id=0,
        )

        self.assertTrue(torch.equal(model_input.condition, torch.zeros_like(model_input.condition)))
        self.assertTrue(torch.equal(model_input.condition_type_ids, torch.zeros_like(model_input.condition_type_ids)))

    def test_build_jit_flow_matching_batch_keeps_condition_when_dropout_is_zero(self) -> None:
        expected_condition = build_edge3d_condition(self.batch)
        model_input = build_jit_flow_matching_batch(
            self.batch,
            t_min=0.0,
            t_max=1.0,
            noise_scale=1.0,
            condition_dropout_p=0.0,
            condition_type_id=0,
        )

        self.assertTrue(torch.equal(model_input.condition, expected_condition))


if __name__ == "__main__":
    unittest.main()