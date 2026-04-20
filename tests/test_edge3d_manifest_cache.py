import json
import tempfile
import unittest
from pathlib import Path

from datasets import build_dataset_from_config


class Edge3DManifestCacheTests(unittest.TestCase):
    def test_build_dataset_creates_reuses_and_overwrites_manifest_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            folder_a = root / "folder_a"
            folder_b = root / "folder_b"
            folder_a.mkdir()
            folder_b.mkdir()

            for path in [
                folder_a / "b_sample.npz",
                folder_a / "d_sample.npz",
                folder_b / "a_sample.npz",
                folder_b / "c_sample.npz",
            ]:
                path.touch()

            manifest_path = root / "manifest_cache" / "edge3d_subset.jsonl"
            cfg = {
                "dataset_type": "edge3d_modalities",
                "data_folders": [str(folder_b), str(folder_a)],
                "manifest_path": str(manifest_path),
                "start_index": 1,
                "data_number": 2,
                "decode_model_normal": True,
            }

            dataset = build_dataset_from_config(cfg)
            self.assertEqual(len(dataset), 2)
            first_manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
            first_records = [json.loads(line) for line in first_manifest_lines]
            self.assertEqual([record["sample_id"] for record in first_records], ["b_sample", "c_sample"])
            self.assertTrue(all(Path(record["tensor_path"]).is_absolute() for record in first_records))
            self.assertEqual(first_records[0]["meta"]["source_folder"], str(folder_a.resolve()))
            self.assertEqual(first_records[1]["meta"]["source_folder"], str(folder_b.resolve()))

            reused_dataset = build_dataset_from_config(
                {
                    **cfg,
                    "start_index": 0,
                    "data_number": 1,
                }
            )
            self.assertEqual(len(reused_dataset), 2)
            self.assertEqual(manifest_path.read_text(encoding="utf-8").splitlines(), first_manifest_lines)

            overwritten_dataset = build_dataset_from_config(
                {
                    **cfg,
                    "overwrite_manifest": True,
                    "start_index": 0,
                    "data_number": 1,
                }
            )
            self.assertEqual(len(overwritten_dataset), 1)
            overwritten_records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["sample_id"] for record in overwritten_records], ["a_sample"])

    def test_build_dataset_raises_when_no_samples_are_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_folder = root / "folder"
            data_folder.mkdir()
            (data_folder / "only_sample.npz").touch()

            with self.assertRaisesRegex(ValueError, "zero samples"):
                build_dataset_from_config(
                    {
                        "dataset_type": "edge3d_modalities",
                        "data_folders": str(data_folder),
                        "manifest_path": str(root / "manifest_cache" / "empty.jsonl"),
                        "start_index": 5,
                        "data_number": 1,
                    }
                )


if __name__ == "__main__":
    unittest.main()