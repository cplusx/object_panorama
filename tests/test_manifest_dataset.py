import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets import ConditionalJiTManifestDataset, conditional_jit_collate_fn
from utils.condition_metadata import LEGACY_CONDITION_TYPE_TO_FLAGS


class ManifestDatasetTests(unittest.TestCase):
    def test_manifest_dataset_loads_npy_and_pt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "input.npy"
            condition_path = root / "condition.pt"
            target_path = root / "target.npy"
            np.save(input_path, np.ones((1, 8, 16), dtype=np.float32))
            torch.save(torch.zeros(5, 8, 16), condition_path)
            np.save(target_path, np.full((1, 8, 16), 2.0, dtype=np.float32))

            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample_a",
                        "input_path": input_path.name,
                        "condition_path": condition_path.name,
                        "target_path": target_path.name,
                        **LEGACY_CONDITION_TYPE_TO_FLAGS[1],
                        "condition_type_id": 1,
                        "meta": {"tag": "demo"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = ConditionalJiTManifestDataset(manifest_path)
            sample = dataset[0]
            self.assertEqual(sample["sample_id"], "sample_a")
            self.assertEqual(tuple(sample["input"].shape), (1, 8, 16))
            self.assertEqual(tuple(sample["condition"].shape), (5, 8, 16))
            self.assertEqual(tuple(sample["target"].shape), (1, 8, 16))
            self.assertEqual(int(sample["condition_type_id"].item()), 1)
            self.assertEqual(sample["meta"]["condition_flags"], LEGACY_CONDITION_TYPE_TO_FLAGS[1])
            self.assertEqual(sample["meta"]["condition_label"], "rgb_plus_edge_depth")

    def test_flags_only_manifest_record_infers_legacy_condition_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tensor_path = root / "tensor.npy"
            np.save(tensor_path, np.ones((1, 4, 4), dtype=np.float32))
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "flags_only",
                        "input_path": tensor_path.name,
                        "condition_path": tensor_path.name,
                        "target_path": tensor_path.name,
                        **LEGACY_CONDITION_TYPE_TO_FLAGS[2],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = ConditionalJiTManifestDataset(manifest_path)
            sample = dataset[0]
            self.assertEqual(int(sample["condition_type_id"].item()), 2)
            self.assertEqual(sample["meta"]["condition_label"], "normal_plus_edge_depth")

    def test_invalid_condition_type_id_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tensor_path = root / "tensor.npy"
            np.save(tensor_path, np.ones((1, 4, 4), dtype=np.float32))
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "sample_id": "bad_sample",
                        "input_path": tensor_path.name,
                        "condition_path": tensor_path.name,
                        "target_path": tensor_path.name,
                        "condition_type_id": 5,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                ConditionalJiTManifestDataset(manifest_path)

    def test_collate_pads_condition_channels(self) -> None:
        batch = [
            {
                "sample_id": "a",
                "input": torch.zeros(1, 8, 16),
                "condition": torch.ones(3, 8, 16),
                "target": torch.zeros(1, 8, 16),
                "condition_type_id": torch.tensor(0, dtype=torch.long),
                "meta": {"condition_flags": LEGACY_CONDITION_TYPE_TO_FLAGS[0], "condition_label": "rgb_plus_normal"},
            },
            {
                "sample_id": "b",
                "input": torch.zeros(1, 8, 16),
                "condition": torch.ones(5, 8, 16),
                "target": torch.zeros(1, 8, 16),
                "condition_type_id": torch.tensor(2, dtype=torch.long),
                "meta": {
                    "condition_flags": LEGACY_CONDITION_TYPE_TO_FLAGS[2],
                    "condition_label": "normal_plus_edge_depth",
                },
            },
        ]
        collated = conditional_jit_collate_fn(batch, max_condition_channels=6)
        self.assertEqual(tuple(collated["condition"].shape), (2, 6, 8, 16))
        self.assertTrue(torch.equal(collated["condition"][0, 3:], torch.zeros(3, 8, 16)))
        self.assertEqual(tuple(collated["condition_type_ids"].shape), (2,))


if __name__ == "__main__":
    unittest.main()