import unittest

import torch

from training import build_paired_supervised_batch, build_x0_prediction_linear_bridge_batch


class TrainingObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.batch = {
            "input": torch.randn(2, 1, 8, 16),
            "condition": torch.randn(2, 5, 8, 16),
            "target": torch.randn(2, 1, 8, 16),
            "condition_type_ids": torch.tensor([0, 2], dtype=torch.long),
        }

    def test_build_paired_supervised_batch(self) -> None:
        model_input = build_paired_supervised_batch(self.batch, fixed_timestep=0.5)
        self.assertEqual(tuple(model_input.sample.shape), (2, 1, 8, 16))
        self.assertEqual(tuple(model_input.condition.shape), (2, 5, 8, 16))
        self.assertTrue(torch.allclose(model_input.timestep, torch.full((2,), 0.5)))
        self.assertTrue(torch.equal(model_input.target, self.batch["target"]))

    def test_build_x0_prediction_linear_bridge_batch(self) -> None:
        model_input = build_x0_prediction_linear_bridge_batch(
            self.batch,
            t_min=0.0,
            t_max=1.0,
            concat_input_to_condition=True,
        )
        self.assertEqual(tuple(model_input.sample.shape), (2, 1, 8, 16))
        self.assertEqual(tuple(model_input.condition.shape), (2, 6, 8, 16))
        self.assertTrue(torch.equal(model_input.target, self.batch["target"]))
        self.assertFalse(torch.allclose(model_input.sample, model_input.target))


if __name__ == "__main__":
    unittest.main()