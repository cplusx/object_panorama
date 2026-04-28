import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.reconstruct_equirectangular_overlap import reconstruct_equirectangular_npz_to_pointclouds


def _build_legacy_model_tensor(resolution: int) -> np.ndarray:
    width = resolution * 2
    model_tensor = np.zeros((35, resolution, width), dtype=np.float32)
    for hit_index in range(5):
        base = hit_index * 7
        model_tensor[base + 0] = 0.1 * (hit_index + 1)
        model_tensor[base + 1] = 0.2 * (hit_index + 1)
        model_tensor[base + 2] = 0.3 * (hit_index + 1)
        model_tensor[base + 3] = 1.0 + 0.25 * hit_index
        model_tensor[base + 4] = -0.5
        model_tensor[base + 5] = 0.0
        model_tensor[base + 6] = 0.5
    return model_tensor


class ReconstructionScriptTests(unittest.TestCase):
    def test_reconstruct_pointcloud_outputs_have_expected_artifact_names(self) -> None:
        resolution = 4
        width = resolution * 2
        edge_tensor = np.stack(
            [
                np.full((resolution, width), 1.5, dtype=np.float32),
                np.full((resolution, width), 2.0, dtype=np.float32),
                np.full((resolution, width), 2.5, dtype=np.float32),
            ],
            axis=0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            npz_path = tmp_path / "synthetic_sample.npz"
            np.savez_compressed(
                npz_path,
                uid=np.asarray("synthetic_uid"),
                model_tensor=_build_legacy_model_tensor(resolution),
                edge_tensor=edge_tensor,
            )

            with mock.patch("tools.reconstruct_equirectangular_overlap._HAS_REFERENCE_STACK", False):
                result = reconstruct_equirectangular_npz_to_pointclouds(
                    npz_path=npz_path,
                    output_dir=tmp_path / "outputs",
                    dataset_root=tmp_path / "dataset_root",
                    skip_reference_metrics=True,
                )

            self.assertEqual(Path(result.model_pointcloud_path).name, "model_points.ply")
            self.assertEqual(Path(result.edge_pointcloud_path).name, "edge_points.ply")
            self.assertEqual(Path(result.overlap_pointcloud_path).name, "overlap_pointcloud.ply")
            self.assertTrue(Path(result.model_pointcloud_path).is_file())
            self.assertTrue(Path(result.edge_pointcloud_path).is_file())
            self.assertTrue(Path(result.overlap_pointcloud_path).is_file())
            self.assertTrue((Path(tmp_dir) / "outputs" / "synthetic_uid" / "synthetic_uid_reconstruction_report.json").is_file())
            self.assertGreater(result.model_points, 0)
            self.assertGreater(result.edge_points, 0)


if __name__ == "__main__":
    unittest.main()