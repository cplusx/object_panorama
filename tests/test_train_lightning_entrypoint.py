import unittest
from argparse import Namespace

from tools.train_lightning_rectangular_conditional_jit import (
    _build_trainer_kwargs,
    _resolve_accumulate_grad_batches,
    _resolve_micro_batches_per_epoch,
)
from utils import load_yaml_config


class TrainLightningEntrypointTests(unittest.TestCase):
    def test_trainer_kwargs_are_epoch_based(self) -> None:
        args = Namespace(device="cuda", strategy=None, limit_train_batches=None, limit_val_batches=None)
        train_cfg = {
            "batch_size": 1,
            "effective_batch_size": 8,
            "num_workers": 0,
            "max_epochs": 100,
            "effective_steps_per_epoch": 2000,
            "train_log_every_n_steps": 10,
        }
        lightning_cfg = {
            "accelerator": "gpu",
            "devices": 2,
            "strategy": "deepspeed_stage_2",
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 1,
        }

        trainer_kwargs = _build_trainer_kwargs(
            args=args,
            train_cfg=train_cfg,
            lightning_cfg=lightning_cfg,
            precision="32-true",
            default_root_dir="./runs/edge3d_flow_train",
            callbacks=[],
            logger=None,
            enable_validation=True,
        )

        self.assertEqual(trainer_kwargs["max_epochs"], 100)
        self.assertEqual(trainer_kwargs["accumulate_grad_batches"], 4)
        self.assertEqual(trainer_kwargs["limit_train_batches"], 8000)
        self.assertEqual(trainer_kwargs["check_val_every_n_epoch"], 1)
        self.assertNotIn("max_steps", trainer_kwargs)
        self.assertNotIn("val_check_interval", trainer_kwargs)

    def test_micro_batches_per_epoch_matches_effective_steps(self) -> None:
        self.assertEqual(_resolve_micro_batches_per_epoch(effective_steps_per_epoch=2000, accumulate_grad_batches=4), 8000)
        self.assertEqual(_resolve_accumulate_grad_batches(batch_size=1, effective_batch_size=8, num_devices=2), 4)

    def test_default_train_config_loads_jit_path(self) -> None:
        config = load_yaml_config("/home/viplab/jiaxin/object_panorama/configs/experiment/edge3d_flow_train.yaml")
        self.assertTrue(config["pretrained"]["load_jit"])
        self.assertEqual(
            config["pretrained"]["public_checkpoint_path"],
            "/home/viplab/jiaxin/object_panorama/pretrained_weights/jit-b-32/checkpoint-last.pth",
        )


if __name__ == "__main__":
    unittest.main()