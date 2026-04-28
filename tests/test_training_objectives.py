import unittest

import torch

from training import build_edge3d_condition, build_jit_flow_matching_batch, compute_prediction_losses


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

    def test_balanced_l2_matches_plain_mse_for_uniform_error(self) -> None:
        target = torch.zeros(1, 3, 2, 4)
        target[:, :, 0, 0] = 1.0
        pred = target + 1.0

        loss_dict = compute_prediction_losses(pred, target, {"name": "balanced_l2"})

        self.assertAlmostEqual(float(loss_dict["loss_total"]), 1.0, places=6)
        self.assertAlmostEqual(float(loss_dict["loss_balanced_l2"]), 1.0, places=6)
        self.assertAlmostEqual(float(loss_dict["loss_edge_l2"]), 1.0, places=6)
        self.assertAlmostEqual(float(loss_dict["loss_non_edge_l2"]), 1.0, places=6)
        self.assertAlmostEqual(float(loss_dict["edge_pixel_fraction"]), 0.125, places=6)
        self.assertAlmostEqual(float(loss_dict["non_edge_pixel_fraction"]), 0.875, places=6)
        self.assertAlmostEqual(float(loss_dict["edge_weight_scale"]), 4.0, places=6)
        self.assertAlmostEqual(float(loss_dict["non_edge_weight_scale"]), 4.0 / 7.0, places=6)

    def test_balanced_l2_is_computed_per_sample(self) -> None:
        target = torch.zeros(2, 3, 2, 2)
        target[0, :, 0, 0] = 1.0
        target[1, :, 0, :] = 1.0

        pred = torch.zeros_like(target)
        pred[0, :, 0, 0] = 2.0
        pred[0, :, 0, 1] = 1.0
        pred[0, :, 1, :] = 1.0
        pred[1, :, 0, :] = 3.0
        pred[1, :, 1, :] = 1.0

        loss_dict = compute_prediction_losses(pred, target, {"name": "balanced_l2"})

        sample0_edge_l2 = 1.0
        sample0_non_edge_l2 = 1.0
        sample1_edge_l2 = 4.0
        sample1_non_edge_l2 = 1.0
        expected_total = 0.5 * ((0.5 * (sample0_edge_l2 + sample0_non_edge_l2)) + (0.5 * (sample1_edge_l2 + sample1_non_edge_l2)))

        self.assertAlmostEqual(float(loss_dict["loss_total"]), expected_total, places=6)
        self.assertAlmostEqual(float(loss_dict["loss_edge_l2"]), 2.5, places=6)
        self.assertAlmostEqual(float(loss_dict["loss_non_edge_l2"]), 1.0, places=6)

    def test_weighted_sum_loss_remains_available(self) -> None:
        pred = torch.tensor([[[[1.0]]]])
        target = torch.tensor([[[[0.0]]]])

        loss_dict = compute_prediction_losses(pred, target, {"name": "weighted_sum", "mse_weight": 2.0, "l1_weight": 3.0})

        self.assertAlmostEqual(float(loss_dict["loss_mse"]), 1.0, places=6)
        self.assertAlmostEqual(float(loss_dict["loss_l1"]), 1.0, places=6)
        self.assertAlmostEqual(float(loss_dict["loss_total"]), 5.0, places=6)


if __name__ == "__main__":
    unittest.main()