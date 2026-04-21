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
            overlap_scene = trimesh.load(outputs["overlap_pointcloud"], force="scene")
            overlap_model_target_pred_scene = trimesh.load(outputs["overlap_model_target_pred"], force="scene")

            self.assertEqual(len(getattr(model_cloud, "vertices", [])), 0)
            self.assertEqual(len(getattr(pred_cloud, "vertices", [])), 0)
            self.assertEqual(len(getattr(target_cloud, "vertices", [])), 0)
            self.assertEqual(len(overlap_scene.geometry), 0)
            self.assertEqual(len(overlap_model_target_pred_scene.geometry), 0)


if __name__ == "__main__":
    unittest.main()