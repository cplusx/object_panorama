import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from edge3d_tensor_format import save_mixed_precision_sample
from datasets import build_dataloader_from_config, build_dataset_from_config
from training.datamodule import RectangularConditionalJiTDataModule, _select_validation_indices


class DataModuleTests(unittest.TestCase):
    def test_datamodule_builds_train_and_val_dataloaders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.jsonl"

            records = []
            for index in range(4):
                tensor_path = root / f"sample_{index}.npz"
                model_tensor = np.ones((14, 8, 16), dtype=np.float32) * (index + 1)
                edge_tensor = np.zeros((3, 8, 16), dtype=np.float32)
                edge_tensor[0] = index + 1
                save_mixed_precision_sample(
                    tensor_path,
                    uid=f"sample_{index}",
                    model_tensor=model_tensor,
                    edge_tensor=edge_tensor,
                    resolution=8,
                    model_max_hits=2,
                    edge_max_hits=3,
                )
                records.append(
                    {
                        "sample_id": f"sample_{index}",
                        "tensor_path": tensor_path.name,
                        "meta": {"split": "train"},
                    }
                )

            manifest_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            data_cfg = {"dataset_type": "edge3d_modalities", "manifest_path": str(manifest_path), "root_dir": str(root)}
            datamodule = RectangularConditionalJiTDataModule(
                train_data_cfg=data_cfg,
                val_data_cfg=data_cfg,
                train_cfg={"batch_size": 2, "num_workers": 0},
                validation_cfg={"num_val_samples": 4},
                max_condition_channels=20,
            )
            datamodule.setup(stage="fit")

            train_batch = next(iter(datamodule.train_dataloader()))
            val_batch = next(iter(datamodule.val_dataloader()))

            self.assertEqual(tuple(train_batch["model_rgb"].shape), (2, 6, 8, 16))
            self.assertEqual(tuple(train_batch["model_depth"].shape), (2, 2, 8, 16))
            self.assertEqual(tuple(train_batch["model_normal"].shape), (2, 6, 8, 16))
            self.assertEqual(tuple(val_batch["edge_depth"].shape), (2, 3, 8, 16))

    def test_dataloader_uses_modality_collate_for_subset_wrapped_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.jsonl"

            records = []
            for index in range(2):
                tensor_path = root / f"sample_{index}.npz"
                model_tensor = np.ones((14, 8, 16), dtype=np.float32) * (index + 1)
                edge_tensor = np.zeros((3, 8, 16), dtype=np.float32)
                edge_tensor[0] = index + 1
                save_mixed_precision_sample(
                    tensor_path,
                    uid=f"sample_{index}",
                    model_tensor=model_tensor,
                    edge_tensor=edge_tensor,
                    resolution=8,
                    model_max_hits=2,
                    edge_max_hits=3,
                )
                records.append(
                    {
                        "sample_id": f"sample_{index}",
                        "tensor_path": str(tensor_path),
                        "meta": {"split": "train"},
                    }
                )

            manifest_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            dataset = build_dataset_from_config(
                {"dataset_type": "edge3d_modalities", "manifest_path": str(manifest_path), "decode_model_normal": True}
            )
            subset = Subset(dataset, [0, 1])
            dataloader = build_dataloader_from_config(
                {"batch_size": 2, "num_workers": 0, "shuffle": False, "drop_last": False},
                subset,
                max_condition_channels=20,
            )

            batch = next(iter(dataloader))
            self.assertIn("model_rgb", batch)
            self.assertIn("model_depth", batch)
            self.assertIn("model_normal", batch)
            self.assertIn("edge_depth", batch)

    def test_validation_indices_are_globally_limited(self) -> None:
        rank0 = _select_validation_indices(num_val_samples=4, rank=0, world_size=2)
        rank1 = _select_validation_indices(num_val_samples=4, rank=1, world_size=2)
        self.assertEqual(rank0, [0, 1])
        self.assertEqual(rank1, [2, 3])
        self.assertEqual(len(rank0) + len(rank1), 4)

    def test_validation_indices_allow_uneven_last_rank(self) -> None:
        rank0 = _select_validation_indices(num_val_samples=5, rank=0, world_size=2)
        rank1 = _select_validation_indices(num_val_samples=5, rank=1, world_size=2)
        self.assertEqual(rank0, [0, 1, 2])
        self.assertEqual(rank1, [3, 4])
        self.assertEqual(len(rank0) + len(rank1), 5)


if __name__ == "__main__":
    unittest.main()