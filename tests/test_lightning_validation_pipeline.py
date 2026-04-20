import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import trimesh

from training.lightning_module import RectangularConditionalJiTLightningModule


class _FakePipeline:
    last_inference_dtype = None

    def __init__(self, model, objective_cfg, inference_dtype="float16") -> None:
        self.model = model
        self.objective_cfg = objective_cfg
        type(self).last_inference_dtype = inference_dtype

    def generate(self, batch, num_steps=20, return_intermediates=False):
        del return_intermediates
        pred = torch.full_like(batch["edge_depth"], 0.5)
        return {
            "pred_edge_depth": pred,
            "initial_noise": torch.zeros_like(pred),
            "num_steps": int(num_steps),
            "condition_channels": 20,
            "effective_inference_dtype": "float16",
        }


class LightningValidationPipelineTests(unittest.TestCase):
    def test_validation_step_uses_pipeline_metrics_and_writes_preview(self) -> None:
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
                "condition_channels_per_type": (20,),
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
            loss_cfg={"mse_weight": 1.0, "l1_weight": 0.5},
            optim_cfg={
                "lr": 1.0e-4,
                "backbone_lr_mult": 0.1,
                "weight_decay": 0.0,
                "optimizer": "adamw",
                "max_epochs": 1,
                "effective_steps_per_epoch": 10,
                "visualize_every_n_steps": 0,
                "lr_scheduler": {"name": "constant"},
            },
            freeze_cfg={},
            pretrained_cfg={},
            validation_cfg={
                "num_inference_steps": 4,
                "inference_dtype": "float16",
                "save_preview_every_n_epochs": 1,
                "save_reconstruction": True,
            },
        )

        batch = {
            "sample_ids": ["sample-a", "sample-b"],
            "model_rgb": torch.randn(2, 15, 64, 128),
            "model_depth": torch.randn(2, 5, 64, 128),
            "model_normal": torch.randn(2, 15, 64, 128),
            "edge_depth": torch.randn(2, 3, 64, 128),
            "meta": [{}, {}],
        }

        logged = []

        def capture_log(name, value, **kwargs):
            del kwargs
            logged.append((name, float(value.detach().cpu())))

        module.log = capture_log

        with tempfile.TemporaryDirectory() as tmp_dir:
            module._trainer = SimpleNamespace(world_size=1, default_root_dir=tmp_dir, global_step=1, current_epoch=0)
            with mock.patch("training.lightning_module.Edge3DX0BridgePipeline", _FakePipeline):
                output = module.validation_step(batch, batch_idx=0)

            self.assertIn("val/infer_loss_total", output)
            self.assertIn("val/infer_mse", output)
            self.assertIn("val/infer_l1", output)

            logged_names = {name for name, _ in logged}
            self.assertEqual(logged_names, {"val/infer_loss_total", "val/infer_mse", "val/infer_l1"})
            self.assertEqual(_FakePipeline.last_inference_dtype, "float16")

            sample_dir = Path(tmp_dir) / "validation_outputs" / "epoch_000001" / "sample-a"
            self.assertTrue((sample_dir / "preview.png").is_file())
            self.assertTrue((sample_dir / "model_points.ply").is_file())
            self.assertTrue((sample_dir / "pred_edge_depth.pt").is_file())
            self.assertTrue((sample_dir / "target_edge_depth.pt").is_file())
            self.assertTrue((sample_dir / "pred_edge_points.ply").is_file())
            self.assertTrue((sample_dir / "target_edge_points.ply").is_file())
            self.assertTrue((sample_dir / "overlap_pointcloud.glb").is_file())
            self.assertTrue((sample_dir / "overlap_model_target_pred.glb").is_file())

            model_cloud = trimesh.load(sample_dir / "model_points.ply", force="mesh")
            pred_cloud = trimesh.load(sample_dir / "pred_edge_points.ply", force="mesh")
            target_cloud = trimesh.load(sample_dir / "target_edge_points.ply", force="mesh")
            overlap_scene = trimesh.load(sample_dir / "overlap_pointcloud.glb", force="scene")
            overlap_model_target_pred_scene = trimesh.load(sample_dir / "overlap_model_target_pred.glb", force="scene")
            self.assertEqual(len(getattr(model_cloud, "faces", [])), 0)
            self.assertEqual(len(getattr(pred_cloud, "faces", [])), 0)
            self.assertEqual(len(getattr(target_cloud, "faces", [])), 0)
            for geometry in overlap_scene.geometry.values():
                self.assertEqual(len(getattr(geometry, "faces", [])), 0)
                self.assertEqual(len(getattr(geometry, "entities", [])), 0)
            self.assertEqual(len(overlap_model_target_pred_scene.geometry), 3)
            for geometry in overlap_model_target_pred_scene.geometry.values():
                self.assertEqual(len(getattr(geometry, "faces", [])), 0)
                self.assertEqual(len(getattr(geometry, "entities", [])), 0)


if __name__ == "__main__":
    unittest.main()