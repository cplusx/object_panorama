import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


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

    def _make_model(self, interaction_mode: str) -> RectangularConditionalJiTModel:
        return RectangularConditionalJiTModel(
            input_size=(64, 128),
            patch_size=16,
            image_in_channels=1,
            image_out_channels=2,
            hidden_size=64,
            depth=4,
            num_heads=4,
            bottleneck_dim=16,
            condition_size=(64, 128),
            condition_channels_per_type=(5, 6, 7),
            cond_base_channels=8,
            cond_bottleneck_dim=12,
            cond_tower_depth=2,
            interaction_mode=interaction_mode,
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
        )
        with torch.no_grad():
            for block in tower.blocks:
                block.adaLN_modulation[-1].weight.normal_(mean=0.0, std=0.02)
                block.adaLN_modulation[-1].bias.zero_()

        condition = torch.randn(2, 8, 64, 128, requires_grad=True)
        timestep_embedding = torch.randn(2, 64, requires_grad=True)
        global_tokens, condition_tokens, sequence = tower(condition, timestep_embedding)
        self.assertEqual(tuple(global_tokens.shape), (2, 1, 64))
        self.assertEqual(tuple(condition_tokens.shape), (2, 32, 64))
        self.assertEqual(tuple(sequence.shape), (2, 33, 64))

        dummy = torch.randn(2, 4, 33, 16)
        prefix_before = dummy[:, :, :1, :].clone()
        prefix_after = tower.rope(dummy)[:, :, :1, :]
        self.assertTrue(torch.allclose(prefix_before, prefix_after, atol=1e-6, rtol=1e-6))

        sequence.square().mean().backward()
        _assert_nonzero_grad(self, condition.grad, "condition tower input")
        _assert_nonzero_grad(self, timestep_embedding.grad, "condition tower timestep embedding")
        _assert_nonzero_grad(self, tower.g_token.grad, "condition tower g token")

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


if __name__ == "__main__":
    unittest.main()