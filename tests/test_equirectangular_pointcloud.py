import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from reconstruction.equirectangular_pointcloud import export_named_pointclouds_glb, export_point_cloud, save_model_target_pred_pointclouds


class EquirectangularPointcloudTests(unittest.TestCase):
    def test_export_point_cloud_writes_empty_ply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "empty.ply"
            export_point_cloud(np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), output_path)

            self.assertTrue(output_path.is_file())
            cloud = trimesh.load(output_path, force="mesh")
            self.assertEqual(len(getattr(cloud, "vertices", [])), 0)
            self.assertEqual(len(getattr(cloud, "faces", [])), 0)

    def test_export_named_pointclouds_glb_writes_empty_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "empty.glb"
            export_named_pointclouds_glb(
                [
                    ("empty-a", np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)),
                    ("empty-b", np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)),
                ],
                output_path,
            )

            self.assertTrue(output_path.is_file())
            scene = trimesh.load(output_path, force="scene")
            self.assertEqual(len(scene.geometry), 0)

    def test_save_model_target_pred_pointclouds_handles_empty_depth_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = save_model_target_pred_pointclouds(
                tmp_dir,
                model_depth=np.zeros((1, 5, 8, 16), dtype=np.float32),
                pred_edge_depth=np.zeros((1, 3, 8, 16), dtype=np.float32),
                target_edge_depth=np.zeros((1, 3, 8, 16), dtype=np.float32),
            )

            for path in outputs.values():
                self.assertTrue(Path(path).is_file())

            model_cloud = trimesh.load(outputs["model_points"], force="mesh")
            pred_cloud = trimesh.load(outputs["pred_edge_points"], force="mesh")
            target_cloud = trimesh.load(outputs["target_edge_points"], force="mesh")
            target_pred_cloud = trimesh.load(outputs["target_pred_points"], force="mesh")
            model_target_pred_cloud = trimesh.load(outputs["model_target_pred_points"], force="mesh")

            self.assertEqual(len(getattr(model_cloud, "vertices", [])), 0)
            self.assertEqual(len(getattr(pred_cloud, "vertices", [])), 0)
            self.assertEqual(len(getattr(target_cloud, "vertices", [])), 0)
            self.assertEqual(len(getattr(target_pred_cloud, "vertices", [])), 0)
            self.assertEqual(len(getattr(model_target_pred_cloud, "vertices", [])), 0)

    def test_save_model_target_pred_pointclouds_filters_near_zero_prediction_noise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_depth = np.full((1, 1, 4, 8), 1.0, dtype=np.float32)
            target_edge_depth = np.full((1, 1, 4, 8), 0.8, dtype=np.float32)
            pred_edge_depth = np.zeros((1, 1, 4, 8), dtype=np.float32)
            pred_edge_depth[0, 0, 0, 0] = 1.0
            pred_edge_depth[0, 0, 1, 1] = 1.0e-5

            outputs = save_model_target_pred_pointclouds(
                tmp_dir,
                model_depth=model_depth,
                pred_edge_depth=pred_edge_depth,
                target_edge_depth=target_edge_depth,
            )

            pred_cloud = trimesh.load(outputs["pred_edge_points"])
            self.assertEqual(len(getattr(pred_cloud, "vertices", [])), 1)


if __name__ == "__main__":
    unittest.main()