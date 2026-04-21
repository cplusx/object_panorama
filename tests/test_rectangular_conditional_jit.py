import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import (
    ConditionStemCollection,
    ConditionTower,
    FullJointMMDiTAdapter,
    ImageInputAdapter,
    ImageOutputAdapter,
    RectangularConditionalJiTModel,
    RectangularVisionRotaryEmbeddingFast,
    SparseCrossAttnAdapter,
)
from models.jit_model import JiTModel


def _assert_nonzero_grad(test_case: unittest.TestCase, tensor: torch.Tensor | None, name: str) -> None:
    test_case.assertIsNotNone(tensor, msg=f"{name} grad is missing")
    test_case.assertTrue(torch.isfinite(tensor).all().item(), msg=f"{name} grad is not finite")
    test_case.assertGreater(float(tensor.abs().sum().item()), 0.0, msg=f"{name} grad is zero")


class RectangularConditionalJiTTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _make_source_jit(self) -> JiTModel:
        return JiTModel(
            input_size=64,
            patch_size=16,
            in_channels=3,
            hidden_size=64,
            depth=4,
            num_heads=4,
            bottleneck_dim=16,
            in_context_len=0,
            in_context_start=0,
        )

    def _make_model(self, interaction_mode: str, **overrides) -> RectangularConditionalJiTModel:
        config = {
            "input_size": (64, 128),
            "patch_size": 16,
            "image_in_channels": 1,
            "image_out_channels": 2,
            "hidden_size": 64,
            "depth": 4,
            "num_heads": 4,
            "bottleneck_dim": 16,
            "condition_size": (64, 128),
            "condition_channels_per_type": (5, 6, 7),
            "cond_base_channels": 8,
            "cond_bottleneck_dim": 12,
            "cond_tower_depth": 2,
            "interaction_mode": interaction_mode,
            "interaction_layers": (1, 3),
            "preset_name": None,
        }
        config.update(overrides)
        return RectangularConditionalJiTModel(**config)

    def _activate_conditioning_path(self, model: RectangularConditionalJiTModel) -> None:
        with torch.no_grad():
            for block in model.blocks:
                block.adaLN_modulation[-1].weight.normal_(mean=0.0, std=0.02)
                block.adaLN_modulation[-1].bias.zero_()
            model.final_layer.adaLN_modulation[-1].weight.normal_(mean=0.0, std=0.02)
            model.final_layer.adaLN_modulation[-1].bias.zero_()
            model.final_layer.linear.weight.normal_(mean=0.0, std=0.02)
            model.final_layer.linear.bias.zero_()

    def test_image_adapters_forward_and_grad(self) -> None:
        image_input = ImageInputAdapter(1)
        x = torch.randn(2, 1, 32, 64, requires_grad=True)
        y = image_input(x)
        self.assertEqual(tuple(y.shape), (2, 3, 32, 64))
        y.mean().backward()
        _assert_nonzero_grad(self, x.grad, "image input")
        _assert_nonzero_grad(self, image_input.proj.weight.grad, "image input adapter weight")

        image_output = ImageOutputAdapter(2)
        z_in = torch.randn(2, 3, 32, 64, requires_grad=True)
        z = image_output(z_in)
        self.assertEqual(tuple(z.shape), (2, 2, 32, 64))
        z.mean().backward()
        _assert_nonzero_grad(self, z_in.grad, "image output input")
        _assert_nonzero_grad(self, image_output.proj.weight.grad, "image output adapter weight")

    def test_condition_stem_collection_mixed_types(self) -> None:
        stems = ConditionStemCollection((2, 3, 4), cond_base_channels=8)
        condition = torch.randn(3, 4, 32, 64, requires_grad=True)
        condition_type_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        out = stems(condition, condition_type_ids)
        self.assertEqual(tuple(out.shape), (3, 8, 32, 64))

        out.square().mean().backward()
        _assert_nonzero_grad(self, condition.grad, "condition input")
        for index, stem in enumerate(stems.stems):
            _assert_nonzero_grad(self, stem.net[0].weight.grad, f"condition stem {index} weight")

    def test_condition_stem_collection_matches_stem_output_dtype(self) -> None:
        class _HalfStem(nn.Module):
            def __init__(self, in_channels: int, out_channels: int):
                super().__init__()
                self.in_channels = int(in_channels)
                self.out_channels = int(out_channels)

            def forward(self, condition: torch.Tensor) -> torch.Tensor:
                batch_size, _, height, width = condition.shape
                return torch.ones((batch_size, self.out_channels, height, width), device=condition.device, dtype=torch.float16)

        stems = ConditionStemCollection((2, 3, 4), cond_base_channels=8)
        stems.stems = nn.ModuleList([_HalfStem(2, 8), _HalfStem(3, 8), _HalfStem(4, 8)])
        condition = torch.randn(3, 4, 16, 32, dtype=torch.float32)
        condition_type_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        out = stems(condition, condition_type_ids)

        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (3, 8, 16, 32))

    def test_condition_tower_shapes_prefix_rope_and_grad(self) -> None:
        tower = ConditionTower(
            input_size=(64, 128),
            patch_size=16,
            in_channels=8,
            bottleneck_dim=12,
            hidden_size=64,
            depth=2,
            num_heads=4,
            g_token_count=1,
            num_condition_types=3,
        )
        with torch.no_grad():
            for block in tower.blocks:
                block.adaLN_modulation[-1].weight.normal_(mean=0.0, std=0.02)
                block.adaLN_modulation[-1].bias.zero_()

        condition = torch.randn(1, 8, 64, 128).repeat(2, 1, 1, 1).requires_grad_()
        timestep_embedding = torch.randn(1, 64).repeat(2, 1).requires_grad_()
        condition_type_ids = torch.tensor([0, 1], dtype=torch.long)
        global_tokens, condition_tokens, sequence = tower(condition, timestep_embedding, condition_type_ids)
        self.assertEqual(tuple(global_tokens.shape), (2, 1, 64))
        self.assertEqual(tuple(condition_tokens.shape), (2, 32, 64))
        self.assertEqual(tuple(sequence.shape), (2, 33, 64))
        self.assertFalse(torch.allclose(sequence[0], sequence[1]))

        dummy = torch.randn(2, 4, 33, 16)
        prefix_before = dummy[:, :, :1, :].clone()
        prefix_after = tower.rope(dummy)[:, :, :1, :]
        self.assertTrue(torch.allclose(prefix_before, prefix_after, atol=1e-6, rtol=1e-6))

        sequence.square().mean().backward()
        _assert_nonzero_grad(self, condition.grad, "condition tower input")
        _assert_nonzero_grad(self, timestep_embedding.grad, "condition tower timestep embedding")
        _assert_nonzero_grad(self, tower.g_token.grad, "condition tower g token")
        _assert_nonzero_grad(self, tower.condition_type_embedding.weight.grad, "condition tower type embedding")

    def test_sparse_cross_attention_adapter_zero_init_and_branch_grad(self) -> None:
        adapter = SparseCrossAttnAdapter(hidden_size=64, num_heads=4)
        self.assertEqual(float(adapter.alpha.detach().item()), 0.0)

        image_rope = RectangularVisionRotaryEmbeddingFast(dim=8, pt_seq_len=(4, 8))
        condition_rope = RectangularVisionRotaryEmbeddingFast(dim=8, pt_seq_len=(4, 8), num_prefix_tokens=1)
        image_tokens = torch.randn(2, 32, 64)
        condition_tokens = torch.randn(2, 33, 64)

        with torch.no_grad():
            out = adapter(image_tokens, condition_tokens, image_rope, condition_rope)
            self.assertTrue(torch.allclose(out, image_tokens, atol=1e-6, rtol=1e-6))

        image_tokens = torch.randn(2, 32, 64, requires_grad=True)
        condition_tokens = torch.randn(2, 33, 64, requires_grad=True)
        out = adapter(image_tokens, condition_tokens, image_rope, condition_rope)
        out.sum().backward()
        _assert_nonzero_grad(self, adapter.alpha.grad, "sparse adapter alpha")

        adapter.zero_grad(set_to_none=True)
        image_tokens = torch.randn(2, 32, 64, requires_grad=True)
        condition_tokens = torch.randn(2, 33, 64, requires_grad=True)
        with torch.no_grad():
            adapter.alpha.fill_(1.0)
        out = adapter(image_tokens, condition_tokens, image_rope, condition_rope)
        out.square().mean().backward()
        _assert_nonzero_grad(self, condition_tokens.grad, "sparse adapter condition input")
        _assert_nonzero_grad(self, adapter.q_proj.weight.grad, "sparse adapter q_proj")
        _assert_nonzero_grad(self, adapter.k_proj.weight.grad, "sparse adapter k_proj")
        _assert_nonzero_grad(self, adapter.out_proj.weight.grad, "sparse adapter out_proj")

    def test_full_joint_mmdit_adapter_zero_init_and_branch_grad(self) -> None:
        adapter = FullJointMMDiTAdapter(hidden_size=64, num_heads=4)
        self.assertEqual(float(adapter.alpha.detach().item()), 0.0)
        self.assertEqual(float(adapter.beta.detach().item()), 0.0)

        image_rope = RectangularVisionRotaryEmbeddingFast(dim=8, pt_seq_len=(4, 8))
        condition_rope = RectangularVisionRotaryEmbeddingFast(dim=8, pt_seq_len=(4, 8), num_prefix_tokens=1)
        image_tokens = torch.randn(2, 32, 64)
        condition_tokens = torch.randn(2, 33, 64)

        with torch.no_grad():
            image_out, condition_out = adapter(image_tokens, condition_tokens, image_rope, condition_rope)
            self.assertTrue(torch.allclose(image_out, image_tokens, atol=1e-6, rtol=1e-6))
            self.assertEqual(tuple(condition_out.shape), tuple(condition_tokens.shape))

        image_tokens = torch.randn(2, 32, 64, requires_grad=True)
        condition_tokens = torch.randn(2, 33, 64, requires_grad=True)
        image_out, _ = adapter(image_tokens, condition_tokens, image_rope, condition_rope)
        image_out.sum().backward()
        _assert_nonzero_grad(self, adapter.alpha.grad, "full joint adapter alpha")
        _assert_nonzero_grad(self, adapter.beta.grad, "full joint adapter beta")

        adapter.zero_grad(set_to_none=True)
        image_tokens = torch.randn(2, 32, 64, requires_grad=True)
        condition_tokens = torch.randn(2, 33, 64, requires_grad=True)
        with torch.no_grad():
            adapter.alpha.fill_(1.0)
            adapter.beta.fill_(1.0)
        image_out, condition_out = adapter(image_tokens, condition_tokens, image_rope, condition_rope)
        (image_out.square().mean() + condition_out.square().mean()).backward()
        _assert_nonzero_grad(self, image_tokens.grad, "full joint image input")
        _assert_nonzero_grad(self, condition_tokens.grad, "full joint condition input")
        _assert_nonzero_grad(self, adapter.image_qkv.weight.grad, "full joint image qkv")
        _assert_nonzero_grad(self, adapter.cond_qkv.weight.grad, "full joint cond qkv")
        _assert_nonzero_grad(self, adapter.image_mlp.w12.weight.grad, "full joint image mlp")

    def test_backbone_transplant_and_sparse_model_grad(self) -> None:
        source_model = self._make_source_jit()
        model = self._make_model("sparse_xattn")
        report = model.load_pretrained_jit_backbone(source_model)
        self.assertEqual(report["copied_modules"], ["t_embedder", "x_embedder", "blocks", "final_layer"])
        self.assertTrue(torch.allclose(model.x_embedder.proj1.weight, source_model.x_embedder.proj1.weight))
        self.assertTrue(torch.allclose(model.blocks[0].attn.qkv.weight, source_model.blocks[0].attn.qkv.weight))

        self._activate_conditioning_path(model)
        first_adapter = model.interaction_blocks["1"]
        with torch.no_grad():
            first_adapter.alpha.fill_(1.0)

        sample = torch.randn(2, 1, 64, 128, requires_grad=True)
        condition = torch.randn(2, 7, 64, 128, requires_grad=True)
        outputs = model(
            sample=sample,
            timestep=torch.tensor([0.25, 0.75]),
            condition=condition,
            condition_type_ids=torch.tensor([0, 2], dtype=torch.long),
            return_intermediates=True,
        )
        self.assertEqual(tuple(outputs.sample.shape), (2, 2, 64, 128))
        self.assertEqual(tuple(outputs.image_tokens.shape), (2, 32, 64))
        self.assertEqual(tuple(outputs.condition_tokens.shape), (2, 33, 64))
        self.assertEqual(tuple(outputs.global_tokens.shape), (2, 1, 64))

        target = torch.randn_like(outputs.sample)
        loss = F.mse_loss(outputs.sample, target)
        loss.backward()

        _assert_nonzero_grad(self, sample.grad, "sparse model sample input")
        _assert_nonzero_grad(self, condition.grad, "sparse model condition input")
        _assert_nonzero_grad(self, model.image_input_adapter.proj.weight.grad, "sparse model input adapter")
        _assert_nonzero_grad(self, model.image_output_adapter.proj.weight.grad, "sparse model output adapter")
        _assert_nonzero_grad(self, model.g_proj.mlp[2].weight.grad, "sparse model g_proj")
        _assert_nonzero_grad(self, model.condition_stems.stems[0].net[0].weight.grad, "sparse model condition stem 0")
        _assert_nonzero_grad(self, model.condition_stems.stems[2].net[0].weight.grad, "sparse model condition stem 2")
        _assert_nonzero_grad(self, first_adapter.q_proj.weight.grad, "sparse model interaction q_proj")

    def test_full_joint_model_grad(self) -> None:
        model = self._make_model("full_joint_mmdit")
        self._activate_conditioning_path(model)
        first_adapter = model.interaction_blocks["1"]
        with torch.no_grad():
            first_adapter.alpha.fill_(1.0)
            first_adapter.beta.fill_(1.0)

        sample = torch.randn(2, 1, 64, 128, requires_grad=True)
        condition = torch.randn(2, 7, 64, 128, requires_grad=True)
        outputs = model(
            sample=sample,
            timestep=torch.tensor([0.1, 0.9]),
            condition=condition,
            condition_type_ids=torch.tensor([1, 2], dtype=torch.long),
            return_intermediates=True,
        )
        self.assertEqual(tuple(outputs.sample.shape), (2, 2, 64, 128))
        self.assertEqual(tuple(outputs.image_tokens.shape), (2, 32, 64))
        self.assertEqual(tuple(outputs.condition_tokens.shape), (2, 33, 64))

        target = torch.randn_like(outputs.sample)
        F.l1_loss(outputs.sample, target).backward()

        _assert_nonzero_grad(self, sample.grad, "full model sample input")
        _assert_nonzero_grad(self, condition.grad, "full model condition input")
        _assert_nonzero_grad(self, model.g_proj.mlp[2].weight.grad, "full model g_proj")
        _assert_nonzero_grad(self, model.interaction_blocks["1"].image_qkv.weight.grad, "full model image qkv")
        _assert_nonzero_grad(self, model.interaction_blocks["1"].cond_qkv.weight.grad, "full model cond qkv")

    def test_full_joint_recompute_global_after_joint_updates_conditioning(self) -> None:
        baseline_model = self._make_model("full_joint_mmdit", recompute_global_after_joint=False)
        self._activate_conditioning_path(baseline_model)
        with torch.no_grad():
            baseline_model.g_proj.mlp[2].weight.normal_(mean=0.0, std=0.05)
            baseline_model.g_proj.mlp[2].bias.zero_()
            for adapter in baseline_model.interaction_blocks.values():
                adapter.alpha.fill_(1.0)
                adapter.beta.fill_(1.0)

        recompute_model = self._make_model("full_joint_mmdit", recompute_global_after_joint=True)
        recompute_model.load_state_dict(baseline_model.state_dict())

        sample = torch.randn(2, 1, 64, 128)
        condition = torch.randn(2, 7, 64, 128)
        timestep = torch.tensor([0.2, 0.8])
        condition_type_ids = torch.tensor([0, 2], dtype=torch.long)

        baseline_outputs = baseline_model(
            sample=sample,
            timestep=timestep,
            condition=condition,
            condition_type_ids=condition_type_ids,
            return_intermediates=True,
        )
        recompute_outputs = recompute_model(
            sample=sample,
            timestep=timestep,
            condition=condition,
            condition_type_ids=condition_type_ids,
            return_intermediates=True,
        )

        self.assertFalse(torch.allclose(baseline_outputs.conditioning, recompute_outputs.conditioning))
        self.assertFalse(torch.allclose(baseline_outputs.sample, recompute_outputs.sample))

    def test_load_pretrained_jit_backbone_from_public_checkpoint_mock_chain(self) -> None:
        source_model = self._make_source_jit()
        mock_state_dict = {f"net.{key}": value.detach().clone() for key, value in source_model.state_dict().items()}
        mock_state_dict["net.in_context_posemb"] = torch.zeros(1, 0, source_model.hidden_size)

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "mock_public_jit_checkpoint.pth"
            torch.save({"model_ema1": mock_state_dict}, checkpoint_path)

            model = self._make_model("sparse_xattn")
            report = model.load_pretrained_jit_backbone_from_public_checkpoint(
                checkpoint_path,
                variant="ema1",
                preset_name=None,
            )

        self.assertEqual(report["source"], "public_checkpoint")
        self.assertEqual(report["load_report"]["missing_key_count"], 0)
        self.assertEqual(report["load_report"]["unexpected_key_count"], 1)
        self.assertEqual(report["load_report"]["unexpected_keys"], ["in_context_posemb"])
        self.assertEqual(report["transplant_report"]["copied_modules"], ["t_embedder", "x_embedder", "blocks", "final_layer"])
        self.assertEqual(report["inferred"]["depth"], 4)
        self.assertEqual(report["inferred"]["hidden_size"], 64)
        self.assertEqual(report["inferred"]["patch_size"], 16)
        self.assertEqual(report["inferred"]["in_context_len"], 0)
        self.assertTrue(torch.allclose(model.x_embedder.proj1.weight, source_model.x_embedder.proj1.weight))
        self.assertTrue(torch.allclose(model.blocks[0].attn.qkv.weight, source_model.blocks[0].attn.qkv.weight))

    def test_target_resolution_patch32_integration_forward_backward(self) -> None:
        model = self._make_model(
            "sparse_xattn",
            input_size=(512, 1024),
            patch_size=32,
            hidden_size=32,
            depth=2,
            num_heads=4,
            bottleneck_dim=8,
            condition_size=(512, 1024),
            condition_channels_per_type=(3, 3, 3),
            cond_base_channels=4,
            cond_bottleneck_dim=8,
            cond_tower_depth=1,
            interaction_layers=(1,),
        )
        self._activate_conditioning_path(model)
        with torch.no_grad():
            model.interaction_blocks["1"].alpha.fill_(1.0)

        self.assertEqual(model.x_embedder.grid_size, (16, 32))
        sample = torch.randn(1, 1, 512, 1024, requires_grad=True)
        condition = torch.randn(1, 3, 512, 1024, requires_grad=True)
        outputs = model(
            sample=sample,
            timestep=torch.tensor([0.35]),
            condition=condition,
            condition_type_ids=torch.tensor([2], dtype=torch.long),
            return_intermediates=True,
        )

        self.assertEqual(tuple(outputs.sample.shape), (1, 2, 512, 1024))
        self.assertEqual(tuple(outputs.image_tokens.shape), (1, 512, 32))
        self.assertEqual(tuple(outputs.condition_tokens.shape), (1, 513, 32))
        self.assertEqual(tuple(outputs.global_tokens.shape), (1, 1, 32))

        loss = F.mse_loss(outputs.sample, torch.randn_like(outputs.sample))
        loss.backward()
        _assert_nonzero_grad(self, sample.grad, "target resolution sample input")
        _assert_nonzero_grad(self, condition.grad, "target resolution condition input")
        _assert_nonzero_grad(self, model.image_input_adapter.proj.weight.grad, "target resolution input adapter")
        _assert_nonzero_grad(self, model.image_output_adapter.proj.weight.grad, "target resolution output adapter")
        _assert_nonzero_grad(self, model.condition_stems.stems[2].net[0].weight.grad, "target resolution condition stem")
        _assert_nonzero_grad(self, model.g_proj.mlp[2].weight.grad, "target resolution g_proj")


if __name__ == "__main__":
    unittest.main()