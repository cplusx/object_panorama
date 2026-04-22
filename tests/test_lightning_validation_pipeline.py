import sys
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
    last_cfg_scale = None

    def __init__(self, model, objective_cfg, inference_dtype="float16") -> None:
        self.model = model
        self.objective_cfg = objective_cfg
        type(self).last_inference_dtype = inference_dtype

    def generate(self, batch, num_steps=20, return_intermediates=False, cfg_scale=1.0):
        del return_intermediates
        type(self).last_cfg_scale = float(cfg_scale)
        pred = torch.full_like(batch["edge_depth"], 0.5)
        return {
            "pred_edge_depth": pred,
            "initial_noise": torch.zeros_like(pred),
            "num_steps": int(num_steps),
            "condition_channels": 20,
            "effective_inference_dtype": "float16",
            "cfg_scale": float(cfg_scale),
            "uses_cfg": bool(cfg_scale > 1.0),
        }


class LightningValidationPipelineTests(unittest.TestCase):
    def test_validation_preview_is_skipped_during_sanity_check(self) -> None:
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
                "cfg_scale": 1.5,
                "save_preview_every_n_epochs": 1,
                "save_reconstruction": True,
            },
        )

        batch = {
            "sample_ids": ["sample-a"],
            "model_rgb": torch.randn(1, 15, 64, 128),
            "model_depth": torch.randn(1, 5, 64, 128),
            "model_normal": torch.randn(1, 15, 64, 128),
            "edge_depth": torch.randn(1, 3, 64, 128),
            "meta": [{}],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            module._trainer = SimpleNamespace(
                world_size=1,
                default_root_dir=tmp_dir,
                global_step=0,
                current_epoch=0,
                is_global_zero=True,
                sanity_checking=True,
                logger=SimpleNamespace(experiment=None),
            )
            module._maybe_save_validation_preview(batch, torch.zeros_like(batch["edge_depth"]))
            self.assertFalse((Path(tmp_dir) / "validation_outputs").exists())

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
                "cfg_scale": 1.5,
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

        image_logs = []
        artifact_logs = []

        class FakeExperiment:
            def log(self, payload):
                image_logs.append(payload)

            def log_artifact(self, artifact):
                artifact_logs.append(artifact)

        class FakeWandbImage:
            def __init__(self, path: str) -> None:
                self.path = path

        class FakeWandbArtifact:
            def __init__(self, name: str, type: str) -> None:
                self.name = name
                self.type = type
                self.files = []

            def add_file(self, path: str, name: str | None = None) -> None:
                self.files.append((path, name))

        fake_wandb = SimpleNamespace(Image=FakeWandbImage, Artifact=FakeWandbArtifact)
        fake_logger = SimpleNamespace(experiment=FakeExperiment())

        with tempfile.TemporaryDirectory() as tmp_dir:
            module._trainer = SimpleNamespace(
                world_size=1,
                default_root_dir=tmp_dir,
                global_step=1,
                current_epoch=0,
                is_global_zero=True,
                logger=fake_logger,
            )
            with mock.patch("training.lightning_module.Edge3DX0BridgePipeline", _FakePipeline):
                with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
                    output = module.validation_step(batch, batch_idx=0)

            self.assertIn("val/infer_loss_total", output)
            self.assertIn("val/infer_mse", output)
            self.assertIn("val/infer_l1", output)

            logged_names = {name for name, _ in logged}
            self.assertEqual(logged_names, {"val/infer_loss_total", "val/infer_mse", "val/infer_l1"})
            self.assertEqual(_FakePipeline.last_inference_dtype, "float16")
            self.assertEqual(_FakePipeline.last_cfg_scale, 1.5)

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

        logged_image_keys = set()
        for payload in image_logs:
            logged_image_keys.update(key for key in payload if key != "global_step")
        self.assertEqual(logged_image_keys, {"val/preview/sample-a", "val/preview/sample-b"})
        self.assertTrue(all(payload["global_step"] == 1 for payload in image_logs))

        self.assertEqual(len(artifact_logs), 2)
        artifact_names = {artifact.name for artifact in artifact_logs}
        self.assertEqual(artifact_names, {"validation_3d_epoch_1_sample-a", "validation_3d_epoch_1_sample-b"})
        for artifact in artifact_logs:
            self.assertEqual(artifact.type, "validation_3d")
            self.assertEqual(
                {name for _, name in artifact.files},
                {
                    "model_points.ply",
                    "pred_edge_points.ply",
                    "target_edge_points.ply",
                    "overlap_pointcloud.glb",
                    "overlap_model_target_pred.glb",
                },
            )


if __name__ == "__main__":
    unittest.main()