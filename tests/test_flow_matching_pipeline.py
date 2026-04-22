import unittest
from types import SimpleNamespace

import torch

from pipeline import Edge3DX0BridgePipeline


class _DeterministicConditionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(condition_channels_per_type=(20,))
        self.forward_conditions: list[torch.Tensor] = []

    def forward(self, sample, timestep, condition, condition_type_ids):
        del sample, timestep, condition_type_ids
        self.forward_conditions.append(condition.detach().cpu().clone())
        return SimpleNamespace(sample=condition[:, :3])


class FlowMatchingPipelineTests(unittest.TestCase):
    def test_generate_uses_edge3d_condition_and_restores_training_state(self) -> None:
        model = _DeterministicConditionModel()
        model.train(True)
        pipeline = Edge3DX0BridgePipeline(
            model,
            {
                "condition_type_id": 0,
                "use_model_rgb": False,
                "use_model_depth": True,
                "use_model_normal": True,
            },
            inference_dtype="float16",
        )

        batch = {
            "sample_ids": ["sample-a"],
            "model_rgb": torch.zeros(1, 15, 4, 8),
            "model_depth": torch.stack(
                [
                    torch.full((1, 4, 8), 1.0),
                    torch.full((1, 4, 8), 2.0),
                    torch.full((1, 4, 8), 3.0),
                    torch.full((1, 4, 8), 4.0),
                    torch.full((1, 4, 8), 5.0),
                ],
                dim=1,
            ),
            "model_normal": torch.zeros(1, 15, 4, 8),
            "edge_depth": torch.zeros(1, 3, 4, 8),
            "meta": [{}],
        }
        noise = torch.zeros(1, 3, 4, 8)

        output = pipeline.generate(batch, num_steps=4, noise=noise, return_intermediates=True, cfg_scale=1.0)

        expected = torch.stack(
            [
                torch.full((4, 8), 1.0),
                torch.full((4, 8), 2.0),
                torch.full((4, 8), 3.0),
            ],
            dim=0,
        ).unsqueeze(0)
        self.assertTrue(torch.equal(output["pred_edge_depth"], expected))
        self.assertTrue(torch.equal(output["initial_noise"], noise))
        self.assertEqual(output["num_steps"], 4)
        self.assertEqual(output["condition_channels"], 20)
        self.assertEqual(output["effective_inference_dtype"], "float32")
        self.assertEqual(output["cfg_scale"], 1.0)
        self.assertFalse(output["uses_cfg"])
        self.assertEqual(output["pred_edge_depth"].dtype, torch.float32)
        self.assertEqual(len(output["intermediates"]), 4)
        self.assertEqual(len(model.forward_conditions), 4)
        self.assertTrue(model.training)

    def test_generate_uses_cfg_with_zero_condition_branch(self) -> None:
        model = _DeterministicConditionModel()
        pipeline = Edge3DX0BridgePipeline(
            model,
            {
                "condition_type_id": 0,
                "use_model_rgb": False,
                "use_model_depth": True,
                "use_model_normal": True,
            },
            inference_dtype="float32",
        )

        batch = {
            "sample_ids": ["sample-a"],
            "model_rgb": torch.zeros(1, 15, 4, 8),
            "model_depth": torch.stack(
                [
                    torch.full((1, 4, 8), 1.0),
                    torch.full((1, 4, 8), 2.0),
                    torch.full((1, 4, 8), 3.0),
                    torch.full((1, 4, 8), 4.0),
                    torch.full((1, 4, 8), 5.0),
                ],
                dim=1,
            ),
            "model_normal": torch.zeros(1, 15, 4, 8),
            "edge_depth": torch.zeros(1, 3, 4, 8),
            "meta": [{}],
        }

        output = pipeline.generate(batch, num_steps=3, noise=torch.zeros(1, 3, 4, 8), cfg_scale=2.0)

        expected_cond = torch.stack(
            [
                torch.full((4, 8), 1.0),
                torch.full((4, 8), 2.0),
                torch.full((4, 8), 3.0),
            ],
            dim=0,
        ).unsqueeze(0)
        expected_condition = torch.cat(
            [
                torch.stack(
                    [
                        torch.full((1, 4, 8), 1.0),
                        torch.full((1, 4, 8), 2.0),
                        torch.full((1, 4, 8), 3.0),
                        torch.full((1, 4, 8), 4.0),
                        torch.full((1, 4, 8), 5.0),
                    ],
                    dim=1,
                ),
                torch.zeros(1, 15, 4, 8),
            ],
            dim=1,
        )
        self.assertTrue(torch.equal(output["pred_edge_depth"], expected_cond * 2.0))
        self.assertEqual(len(model.forward_conditions), 6)
        self.assertTrue(torch.equal(model.forward_conditions[0], expected_condition))
        self.assertTrue(torch.equal(model.forward_conditions[1], torch.zeros_like(model.forward_conditions[1])))
        self.assertEqual(model.forward_conditions[0].shape, model.forward_conditions[1].shape)
        self.assertEqual(output["cfg_scale"], 2.0)
        self.assertTrue(output["uses_cfg"])


if __name__ == "__main__":
    unittest.main()