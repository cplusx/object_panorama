import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from condition_metadata import LEGACY_CONDITION_TYPE_TO_FLAGS
from training.datamodule import RectangularConditionalJiTDataModule


class DataModuleTests(unittest.TestCase):
    def test_datamodule_builds_train_and_val_dataloaders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.jsonl"

            records = []
            for index in range(4):
                input_path = root / f"input_{index}.npy"
                condition_path = root / f"condition_{index}.npy"
                target_path = root / f"target_{index}.npy"
                np.save(input_path, np.ones((1, 8, 16), dtype=np.float32))
                np.save(condition_path, np.ones((5, 8, 16), dtype=np.float32))
                np.save(target_path, np.zeros((1, 8, 16), dtype=np.float32))
                records.append(
                    {
                        "sample_id": f"sample_{index}",
                        "input_path": input_path.name,
                        "condition_path": condition_path.name,
                        "target_path": target_path.name,
                        **LEGACY_CONDITION_TYPE_TO_FLAGS[index % 3],
                        "condition_type_id": index % 3,
                        "meta": {"split": "train"},
                    }
                )

            manifest_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            data_cfg = {"manifest_path": str(manifest_path), "root_dir": str(root)}
            datamodule = RectangularConditionalJiTDataModule(
                train_data_cfg=data_cfg,
                val_data_cfg=data_cfg,
                train_cfg={"batch_size": 2, "num_workers": 0},
                max_condition_channels=5,
            )
            datamodule.setup(stage="fit")

            train_batch = next(iter(datamodule.train_dataloader()))
            val_batch = next(iter(datamodule.val_dataloader()))

            self.assertEqual(tuple(train_batch["input"].shape), (2, 1, 8, 16))
            self.assertEqual(tuple(train_batch["condition"].shape), (2, 5, 8, 16))
            self.assertEqual(tuple(val_batch["target"].shape), (2, 1, 8, 16))


if __name__ == "__main__":
    unittest.main()