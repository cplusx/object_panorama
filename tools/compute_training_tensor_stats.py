from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset_from_config
from utils import load_yaml_config
from utils.condition_metadata import condition_variant_specs


PERCENTILE_NAMES = ("p01", "p05", "p50", "p95", "p99")
PERCENTILE_VALUES = (1, 5, 50, 95, 99)
CONDITION_VARIANTS = condition_variant_specs()
LEGACY_CONDITION_TYPE_TO_VARIANT = {
    spec["legacy_condition_type_id"]: spec for spec in CONDITION_VARIANTS
}
DEFAULT_EQUIRECTANGULAR_DIR = Path("/home/devdata/edge3d_data/equirectangular_data")
CSV_COLUMNS = [
    "channel_index",
    "channel_sample_count",
    "num_values",
    "num_finite",
    "num_nan",
    "num_posinf",
    "num_neginf",
    "finite_fraction",
    "min",
    "max",
    "mean",
    "std",
    "abs_max",
    "nonzero_fraction",
    "positive_fraction",
    "negative_fraction",
    "p01",
    "p05",
    "p50",
    "p95",
    "p99",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute streaming value-range statistics for the actual tensors used by training."
    )
    parser.add_argument("config", nargs="?", help="Path to either a data config or a full experiment config")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--source", default="auto", choices=["auto", "manifest", "equirectangular"])
    parser.add_argument(
        "--equirectangular-dir",
        default=None,
        help="Optional NPZ directory to scan directly, defaults to /home/devdata/edge3d_data/equirectangular_data",
    )
    parser.add_argument(
        "--condition-preset",
        default="pairwise_non_target_modalities",
        choices=["pairwise_non_target_modalities"],
        help="How to synthesize condition types when scanning raw equirectangular NPZ files.",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to analysis/tensor_stats/<experiment_name>/")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--reservoir-size", type=int, default=100000)
    parser.add_argument(
        "--max-reservoir-update-values",
        type=int,
        default=4096,
        help="Maximum number of finite values sampled from each tensor update before feeding the percentile reservoir.",
    )
    parser.add_argument("--save-hist", action="store_true")
    parser.add_argument("--hist-bins", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--include-raw-modalities",
        action="store_true",
        help="Also compute stats for raw model_rgb/model_depth/model_normal/edge_depth modalities.",
    )
    return parser.parse_args()


class ReservoirSampler:
    def __init__(self, capacity: int, seed: int, max_update_values: int):
        self.capacity = max(int(capacity), 0)
        self.max_update_values = max(int(max_update_values), 0)
        self.rng = np.random.default_rng(seed)
        self.values = np.empty(0, dtype=np.float32)
        self.keys = np.empty(0, dtype=np.float32)

    def update(self, values: np.ndarray) -> None:
        if self.capacity <= 0:
            return
        flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat_values.size == 0:
            return

        if self.max_update_values > 0 and flat_values.size > self.max_update_values:
            sample_indices = self.rng.integers(0, flat_values.size, size=self.max_update_values, endpoint=False)
            flat_values = flat_values[sample_indices]

        self._ingest_sampled_values(flat_values)

    def as_array(self) -> np.ndarray:
        return self.values.copy()

    def merge(self, other: ReservoirSampler) -> None:
        self._ingest_sampled_values(other.values)

    def quantiles(self) -> dict[str, float | None]:
        if self.values.size == 0:
            return {name: None for name in PERCENTILE_NAMES}
        percentile_values = _compute_percentiles(self.values.astype(np.float64, copy=False), PERCENTILE_VALUES)
        return {name: float(value) for name, value in zip(PERCENTILE_NAMES, percentile_values)}

    def _ingest_sampled_values(self, flat_values: np.ndarray) -> None:
        if self.capacity <= 0:
            return
        flat_values = np.asarray(flat_values, dtype=np.float32).reshape(-1)
        if flat_values.size == 0:
            return

        new_keys = self.rng.random(flat_values.size).astype(np.float32, copy=False)
        all_values = np.concatenate((self.values, flat_values), axis=0)
        all_keys = np.concatenate((self.keys, new_keys), axis=0)
        if all_keys.size <= self.capacity:
            self.values = all_values
            self.keys = all_keys
            return

        kth = all_keys.size - self.capacity
        keep_indices = np.argpartition(all_keys, kth)[-self.capacity :]
        self.values = all_values[keep_indices]
        self.keys = all_keys[keep_indices]


@dataclass
class StreamingValueStats:
    reservoir_size: int
    seed: int
    max_reservoir_update_values: int

    def __post_init__(self) -> None:
        self.sampler = ReservoirSampler(self.reservoir_size, self.seed, self.max_reservoir_update_values)
        self.num_values = 0
        self.num_finite = 0
        self.num_nan = 0
        self.num_posinf = 0
        self.num_neginf = 0
        self.num_nonzero = 0
        self.num_positive = 0
        self.num_negative = 0
        self.value_sum = 0.0
        self.value_sumsq = 0.0
        self.min_value = math.inf
        self.max_value = -math.inf
        self.abs_max = 0.0

    def update(self, values: np.ndarray) -> None:
        flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
        if flat_values.size == 0:
            return

        self.num_values += int(flat_values.size)
        self.num_nan += int(np.isnan(flat_values).sum())
        self.num_posinf += int(np.isposinf(flat_values).sum())
        self.num_neginf += int(np.isneginf(flat_values).sum())

        finite_mask = np.isfinite(flat_values)
        finite_count = int(finite_mask.sum())
        self.num_finite += finite_count
        if finite_count == 0:
            return

        finite_values = flat_values[finite_mask].astype(np.float64, copy=False)
        self.value_sum += float(finite_values.sum(dtype=np.float64))
        self.value_sumsq += float(np.dot(finite_values, finite_values))
        self.min_value = min(self.min_value, float(finite_values.min()))
        self.max_value = max(self.max_value, float(finite_values.max()))
        self.abs_max = max(self.abs_max, float(np.abs(finite_values).max()))
        self.num_nonzero += int(np.count_nonzero(finite_values))
        self.num_positive += int(np.count_nonzero(finite_values > 0.0))
        self.num_negative += int(np.count_nonzero(finite_values < 0.0))
        self.sampler.update(finite_values.astype(np.float32, copy=False))

    def update_preaggregated(
        self,
        *,
        num_values: int,
        num_finite: int,
        num_nan: int,
        num_posinf: int,
        num_neginf: int,
        num_nonzero: int,
        num_positive: int,
        num_negative: int,
        value_sum: float,
        value_sumsq: float,
        min_value: float | None,
        max_value: float | None,
        abs_max: float | None,
        sample_values: np.ndarray | None,
    ) -> None:
        self.num_values += int(num_values)
        self.num_finite += int(num_finite)
        self.num_nan += int(num_nan)
        self.num_posinf += int(num_posinf)
        self.num_neginf += int(num_neginf)
        if int(num_finite) == 0:
            return

        self.num_nonzero += int(num_nonzero)
        self.num_positive += int(num_positive)
        self.num_negative += int(num_negative)
        self.value_sum += float(value_sum)
        self.value_sumsq += float(value_sumsq)
        if min_value is not None:
            self.min_value = min(self.min_value, float(min_value))
        if max_value is not None:
            self.max_value = max(self.max_value, float(max_value))
        if abs_max is not None:
            self.abs_max = max(self.abs_max, float(abs_max))
        if sample_values is not None and sample_values.size > 0:
            self.sampler.update(sample_values)

    def summary(self) -> dict[str, Any]:
        finite_fraction = _safe_ratio(self.num_finite, self.num_values)
        if self.num_finite == 0:
            summary = {
                "num_values": int(self.num_values),
                "num_finite": int(self.num_finite),
                "num_nan": int(self.num_nan),
                "num_posinf": int(self.num_posinf),
                "num_neginf": int(self.num_neginf),
                "finite_fraction": finite_fraction,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "abs_max": None,
                "nonzero_fraction": None,
                "positive_fraction": None,
                "negative_fraction": None,
            }
            summary.update(self.sampler.quantiles())
            return summary

        mean_value = self.value_sum / self.num_finite
        variance = max(self.value_sumsq / self.num_finite - mean_value * mean_value, 0.0)
        summary = {
            "num_values": int(self.num_values),
            "num_finite": int(self.num_finite),
            "num_nan": int(self.num_nan),
            "num_posinf": int(self.num_posinf),
            "num_neginf": int(self.num_neginf),
            "finite_fraction": finite_fraction,
            "min": float(self.min_value),
            "max": float(self.max_value),
            "mean": float(mean_value),
            "std": float(math.sqrt(variance)),
            "abs_max": float(self.abs_max),
            "nonzero_fraction": _safe_ratio(self.num_nonzero, self.num_finite),
            "positive_fraction": _safe_ratio(self.num_positive, self.num_finite),
            "negative_fraction": _safe_ratio(self.num_negative, self.num_finite),
        }
        summary.update(self.sampler.quantiles())
        return summary

    def reservoir_values(self) -> np.ndarray:
        return self.sampler.as_array()

    def merge(self, other: StreamingValueStats) -> None:
        self.num_values += int(other.num_values)
        self.num_finite += int(other.num_finite)
        self.num_nan += int(other.num_nan)
        self.num_posinf += int(other.num_posinf)
        self.num_neginf += int(other.num_neginf)
        self.num_nonzero += int(other.num_nonzero)
        self.num_positive += int(other.num_positive)
        self.num_negative += int(other.num_negative)
        self.value_sum += float(other.value_sum)
        self.value_sumsq += float(other.value_sumsq)
        if other.num_finite > 0:
            self.min_value = min(self.min_value, float(other.min_value))
            self.max_value = max(self.max_value, float(other.max_value))
            self.abs_max = max(self.abs_max, float(other.abs_max))
        self.sampler.merge(other.sampler)


class TensorStatsGroup:
    def __init__(self, name: str, reservoir_size: int, seed: int, max_reservoir_update_values: int):
        self.name = name
        self.sample_count = 0
        self.shape_distribution: Counter[str] = Counter()
        self.channel_count_distribution: Counter[int] = Counter()
        self.channel_sample_counts: Counter[int] = Counter()
        self.global_stats = StreamingValueStats(
            reservoir_size=reservoir_size,
            seed=seed,
            max_reservoir_update_values=max_reservoir_update_values,
        )
        self.per_channel: dict[int, StreamingValueStats] = {}
        self._reservoir_size = reservoir_size
        self._seed = seed
        self._max_reservoir_update_values = max_reservoir_update_values
        self._channel_rng = np.random.default_rng(seed + 900000)

    def update(self, tensor: torch.Tensor | np.ndarray) -> None:
        array = _ensure_chw_numpy(tensor)
        self.sample_count += 1
        self.shape_distribution[_shape_key(array.shape)] += 1
        self.channel_count_distribution[int(array.shape[0])] += 1
        self.global_stats.update(array.reshape(-1))

        flat = array.reshape(int(array.shape[0]), -1).astype(np.float32, copy=False)
        finite_mask = np.isfinite(flat)
        safe_values = np.where(finite_mask, flat, 0.0)
        num_values_per_channel = int(flat.shape[1])
        num_nan = np.isnan(flat).sum(axis=1)
        num_posinf = np.isposinf(flat).sum(axis=1)
        num_neginf = np.isneginf(flat).sum(axis=1)
        num_finite = finite_mask.sum(axis=1)
        value_sum = safe_values.sum(axis=1, dtype=np.float64)
        value_sumsq = np.sum(safe_values * safe_values, axis=1, dtype=np.float64)
        min_values = np.where(finite_mask, flat, np.inf).min(axis=1)
        max_values = np.where(finite_mask, flat, -np.inf).max(axis=1)
        abs_max = np.abs(safe_values).max(axis=1)
        num_nonzero = np.count_nonzero(safe_values, axis=1)
        num_positive = np.count_nonzero(safe_values > 0.0, axis=1)
        num_negative = np.count_nonzero(safe_values < 0.0, axis=1)

        sampled = None
        if self._max_reservoir_update_values > 0 and flat.shape[1] > 0:
            sample_count = min(int(flat.shape[1]), int(self._max_reservoir_update_values))
            sample_indices = self._channel_rng.integers(0, flat.shape[1], size=sample_count, endpoint=False)
            sampled = flat[:, sample_indices]

        for channel_index in range(int(array.shape[0])):
            self.channel_sample_counts[channel_index] += 1
            channel_stats = self.per_channel.get(channel_index)
            if channel_stats is None:
                channel_stats = StreamingValueStats(
                    reservoir_size=self._reservoir_size,
                    seed=self._seed + 1000 + channel_index,
                    max_reservoir_update_values=self._max_reservoir_update_values,
                )
                self.per_channel[channel_index] = channel_stats
            channel_sample = None
            if sampled is not None:
                channel_sample = sampled[channel_index]
                channel_sample = channel_sample[np.isfinite(channel_sample)]
            channel_stats.update_preaggregated(
                num_values=num_values_per_channel,
                num_finite=int(num_finite[channel_index]),
                num_nan=int(num_nan[channel_index]),
                num_posinf=int(num_posinf[channel_index]),
                num_neginf=int(num_neginf[channel_index]),
                num_nonzero=int(num_nonzero[channel_index]),
                num_positive=int(num_positive[channel_index]),
                num_negative=int(num_negative[channel_index]),
                value_sum=float(value_sum[channel_index]),
                value_sumsq=float(value_sumsq[channel_index]),
                min_value=None if int(num_finite[channel_index]) == 0 else float(min_values[channel_index]),
                max_value=None if int(num_finite[channel_index]) == 0 else float(max_values[channel_index]),
                abs_max=None if int(num_finite[channel_index]) == 0 else float(abs_max[channel_index]),
                sample_values=channel_sample,
            )

    def summary(self) -> dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "num_channels_observed": int(len(self.per_channel)),
            "shape_distribution": _counter_to_json_dict(self.shape_distribution),
            "channel_count_distribution": _counter_to_json_dict(self.channel_count_distribution),
            "global": self.global_stats.summary(),
        }

    def channel_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for channel_index in sorted(self.per_channel):
            row = {
                "channel_index": int(channel_index),
                "channel_sample_count": int(self.channel_sample_counts[channel_index]),
            }
            row.update(self.per_channel[channel_index].summary())
            rows.append(row)
        return rows

    def merge(self, other: TensorStatsGroup) -> None:
        self.sample_count += int(other.sample_count)
        self.shape_distribution.update(other.shape_distribution)
        self.channel_count_distribution.update(other.channel_count_distribution)
        self.channel_sample_counts.update(other.channel_sample_counts)
        self.global_stats.merge(other.global_stats)
        for channel_index, other_stats in other.per_channel.items():
            channel_stats = self.per_channel.get(channel_index)
            if channel_stats is None:
                channel_stats = StreamingValueStats(
                    reservoir_size=self._reservoir_size,
                    seed=self._seed + 1000 + channel_index,
                    max_reservoir_update_values=self._max_reservoir_update_values,
                )
                self.per_channel[channel_index] = channel_stats
            channel_stats.merge(other_stats)


def main() -> None:
    args = parse_args()
    source = _resolve_source(args)
    if source == "manifest":
        _run_manifest_stats(args)
        return
    _run_equirectangular_stats(args)


def _run_manifest_stats(args: argparse.Namespace) -> None:
    if args.config is None:
        raise ValueError("A config path is required when source=manifest")
    config_path = Path(args.config).expanduser().resolve()
    loaded_config = load_yaml_config(config_path)
    data_cfg, experiment_name = _resolve_data_config(loaded_config, split=args.split, config_path=config_path)
    dataset = build_dataset_from_config(data_cfg)
    total_samples = len(dataset)
    max_samples = total_samples if args.max_samples is None else min(total_samples, int(args.max_samples))
    if max_samples <= 0:
        raise ValueError("Requested sample count is zero; nothing to process")

    output_dir = _resolve_output_dir(args.output_dir, experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, condition_groups = _make_primary_groups(args)

    print(f"Loading dataset from {config_path}")
    print(f"Split: {args.split}")
    print(f"Total samples available: {total_samples}")
    print(f"Samples to process: {max_samples}")
    print(f"Output directory: {output_dir}")
    print("Forced data config override: horizontal_flip_p=0.0")

    for index in range(max_samples):
        sample = dataset[index]
        input_tensor = sample["input"]
        target_tensor = sample["target"]
        condition_tensor = sample["condition"]
        condition_meta = _condition_metadata_from_sample(sample)
        condition_label = condition_meta["label"]

        groups["input"].update(input_tensor)
        groups["target"].update(target_tensor)
        groups["condition_all"].update(condition_tensor)
        condition_groups[condition_label].update(condition_tensor)

        if args.progress_every > 0 and ((index + 1) % args.progress_every == 0 or (index + 1) == max_samples):
            print(f"Processed {index + 1}/{max_samples} samples")

    generated_files = _write_group_outputs(
        output_dir=output_dir,
        groups=groups,
        condition_groups=condition_groups,
        extra_groups={},
        save_hist=args.save_hist,
        hist_bins=args.hist_bins,
    )
    summary = {
        "source": "manifest",
        "config_path": str(config_path),
        "experiment_name": experiment_name,
        "split": args.split,
        "dataset_total_samples": int(total_samples),
        "processed_samples": int(max_samples),
        "reservoir_size": int(args.reservoir_size),
        "max_reservoir_update_values": int(args.max_reservoir_update_values),
        "seed": int(args.seed),
        "output_dir": str(output_dir),
        "data_config": {
            "manifest_path": str(data_cfg["manifest_path"]),
            "root_dir": str(data_cfg.get("root_dir")) if data_cfg.get("root_dir") is not None else None,
            "resize": list(data_cfg["resize"]) if data_cfg.get("resize") is not None else None,
            "horizontal_flip_p": float(data_cfg.get("horizontal_flip_p", 0.0)),
        },
        "input": groups["input"].summary(),
        "target": groups["target"].summary(),
        "condition_all_types_combined": groups["condition_all"].summary(),
        "condition_variant_metadata": {
            spec["label"]: {
                **spec["flags"],
                "legacy_condition_type_id": spec["legacy_condition_type_id"],
            }
            for spec in CONDITION_VARIANTS
        },
        "condition_by_label": {
            spec["label"]: condition_groups[spec["label"]].summary()
            for spec in CONDITION_VARIANTS
        },
        "generated_files": generated_files,
    }
    _write_json(output_dir / "summary.json", summary)
    _print_console_summary(groups, condition_groups)


def _condition_metadata_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample.get("meta")
    if isinstance(meta, dict):
        label = meta.get("condition_label")
        flags = meta.get("condition_flags")
        legacy_condition_type_id = meta.get("legacy_condition_type_id")
        if label is not None and flags is not None:
            if legacy_condition_type_id is None:
                legacy_condition_type_id = int(sample["condition_type_id"])
            return {
                "label": str(label),
                "flags": dict(flags),
                "legacy_condition_type_id": int(legacy_condition_type_id),
            }

    return LEGACY_CONDITION_TYPE_TO_VARIANT[int(sample["condition_type_id"])]


def _run_equirectangular_stats(args: argparse.Namespace) -> None:
    equirectangular_dir = _resolve_equirectangular_dir(args.equirectangular_dir)
    sample_paths = sorted(equirectangular_dir.glob("*.npz"))
    total_samples = len(sample_paths)
    max_samples = total_samples if args.max_samples is None else min(total_samples, int(args.max_samples))
    if max_samples <= 0:
        raise ValueError("Requested sample count is zero; nothing to process")

    mapping = _condition_mapping_from_preset(args.condition_preset)
    experiment_name = f"{equirectangular_dir.name}_{args.condition_preset}"
    output_dir = _resolve_output_dir(args.output_dir, experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_sample_paths = sample_paths[:max_samples]

    print(f"Loading raw NPZ samples from {equirectangular_dir}")
    print(f"Total samples available: {total_samples}")
    print(f"Samples to process: {max_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Condition preset: {args.condition_preset}")
    print(f"Worker count: {args.num_workers}")
    print(f"Include raw modalities: {args.include_raw_modalities}")

    groups, condition_component_groups, extra_groups = _process_equirectangular_samples(
        sample_paths=selected_sample_paths,
        mapping=mapping,
        reservoir_size=args.reservoir_size,
        max_reservoir_update_values=args.max_reservoir_update_values,
        seed=args.seed,
        include_raw_modalities=args.include_raw_modalities,
        num_workers=args.num_workers,
        progress_every=args.progress_every,
    )
    condition_groups = _build_condition_groups_from_components(
        mapping=mapping,
        component_groups=condition_component_groups,
        reservoir_size=args.reservoir_size,
        seed=args.seed,
        max_reservoir_update_values=args.max_reservoir_update_values,
    )
    groups["condition_all"] = _combine_condition_all_group(
        reservoir_size=args.reservoir_size,
        seed=args.seed,
        max_reservoir_update_values=args.max_reservoir_update_values,
        condition_groups=condition_groups,
    )

    generated_files = _write_group_outputs(
        output_dir=output_dir,
        groups=groups,
        condition_groups=condition_groups,
        extra_groups=extra_groups,
        save_hist=args.save_hist,
        hist_bins=args.hist_bins,
    )
    summary = {
        "source": "equirectangular",
        "equirectangular_dir": str(equirectangular_dir),
        "experiment_name": experiment_name,
        "dataset_total_samples": int(total_samples),
        "processed_samples": int(max_samples),
        "reservoir_size": int(args.reservoir_size),
        "max_reservoir_update_values": int(args.max_reservoir_update_values),
        "seed": int(args.seed),
        "output_dir": str(output_dir),
        "input_derivation": "input = nan_to_num(edge_depth[0], nan=0, posinf=0, neginf=0)",
        "target_derivation": "target = nan_to_num(model_depth[0], nan=0, posinf=0, neginf=0)",
        "condition_preset": {
            "name": args.condition_preset,
            "is_inferred": True,
            "condition_variants": {
                label: mapping[label]
                for label in (spec["label"] for spec in CONDITION_VARIANTS)
            },
        },
        "input": groups["input"].summary(),
        "target": groups["target"].summary(),
        "condition_all_types_combined": groups["condition_all"].summary(),
        "condition_by_label": {
            spec["label"]: condition_groups[spec["label"]].summary()
            for spec in CONDITION_VARIANTS
        },
        "generated_files": generated_files,
    }
    if extra_groups:
        if "raw_model_depth_all_hits" in extra_groups:
            extra_groups = {
                "raw_model_rgb": condition_component_groups["model_rgb"],
                "raw_model_normal": condition_component_groups["model_normal"],
                "raw_edge_depth_all_hits": condition_component_groups["edge_depth"],
                **extra_groups,
            }
        summary["raw_modalities"] = {
            group_name: extra_groups[group_name].summary()
            for group_name in sorted(extra_groups)
        }
    _write_json(output_dir / "summary.json", summary)
    _print_console_summary(groups, condition_groups)


def _resolve_source(args: argparse.Namespace) -> str:
    if args.source != "auto":
        return args.source
    if args.equirectangular_dir is not None:
        return "equirectangular"
    if args.config is not None:
        return "manifest"
    return "equirectangular"


def _resolve_equirectangular_dir(path_value: str | None) -> Path:
    if path_value is None:
        return DEFAULT_EQUIRECTANGULAR_DIR.resolve()
    return Path(path_value).expanduser().resolve()


def _make_primary_groups(args: argparse.Namespace) -> tuple[dict[str, TensorStatsGroup], dict[str, TensorStatsGroup]]:
    return _create_primary_groups(
        reservoir_size=args.reservoir_size,
        seed=args.seed,
        max_reservoir_update_values=args.max_reservoir_update_values,
    )


def _create_primary_groups(
    *,
    reservoir_size: int,
    seed: int,
    max_reservoir_update_values: int,
) -> tuple[dict[str, TensorStatsGroup], dict[str, TensorStatsGroup]]:
    groups = {
        "input": TensorStatsGroup(
            "input",
            reservoir_size=reservoir_size,
            seed=seed + 10000,
            max_reservoir_update_values=max_reservoir_update_values,
        ),
        "target": TensorStatsGroup(
            "target",
            reservoir_size=reservoir_size,
            seed=seed + 20000,
            max_reservoir_update_values=max_reservoir_update_values,
        ),
        "condition_all": TensorStatsGroup(
            "condition_all",
            reservoir_size=reservoir_size,
            seed=seed + 30000,
            max_reservoir_update_values=max_reservoir_update_values,
        ),
    }
    condition_groups = {
        spec["label"]: TensorStatsGroup(
            f"condition_{spec['label']}",
            reservoir_size=reservoir_size,
            seed=seed + 40000 + 10000 * spec["legacy_condition_type_id"],
            max_reservoir_update_values=max_reservoir_update_values,
        )
        for spec in CONDITION_VARIANTS
    }
    return groups, condition_groups


def _create_extra_groups(
    *,
    reservoir_size: int,
    seed: int,
    max_reservoir_update_values: int,
    include_raw_modalities: bool,
) -> dict[str, TensorStatsGroup]:
    if not include_raw_modalities:
        return {}
    return {
        "raw_model_depth_all_hits": TensorStatsGroup(
            "raw_model_depth_all_hits",
            reservoir_size=reservoir_size,
            seed=seed + 60000,
            max_reservoir_update_values=max_reservoir_update_values,
        ),
    }


def _create_condition_component_groups(
    *,
    reservoir_size: int,
    seed: int,
    max_reservoir_update_values: int,
) -> dict[str, TensorStatsGroup]:
    return {
        "model_rgb": TensorStatsGroup(
            "condition_component_model_rgb",
            reservoir_size=reservoir_size,
            seed=seed + 50000,
            max_reservoir_update_values=max_reservoir_update_values,
        ),
        "model_normal": TensorStatsGroup(
            "condition_component_model_normal",
            reservoir_size=reservoir_size,
            seed=seed + 70000,
            max_reservoir_update_values=max_reservoir_update_values,
        ),
        "edge_depth": TensorStatsGroup(
            "condition_component_edge_depth",
            reservoir_size=reservoir_size,
            seed=seed + 80000,
            max_reservoir_update_values=max_reservoir_update_values,
        ),
    }


def _combine_condition_all_group(
    *,
    reservoir_size: int,
    seed: int,
    max_reservoir_update_values: int,
    condition_groups: dict[str, TensorStatsGroup],
) -> TensorStatsGroup:
    combined = TensorStatsGroup(
        "condition_all",
        reservoir_size=reservoir_size,
        seed=seed + 30000,
        max_reservoir_update_values=max_reservoir_update_values,
    )
    for spec in CONDITION_VARIANTS:
        combined.merge(condition_groups[spec["label"]])
    return combined


def _build_condition_groups_from_components(
    *,
    mapping: dict[str, dict[str, Any]],
    component_groups: dict[str, TensorStatsGroup],
    reservoir_size: int,
    seed: int,
    max_reservoir_update_values: int,
) -> dict[str, TensorStatsGroup]:
    condition_groups: dict[str, TensorStatsGroup] = {}
    for variant in CONDITION_VARIANTS:
        label = variant["label"]
        spec = mapping[label]
        condition_groups[label] = _build_condition_group_from_components(
            name=f"condition_{label}",
            component_groups=[component_groups[component_name] for component_name in spec["components"]],
            reservoir_size=reservoir_size,
            seed=seed + 40000 + 10000 * spec["legacy_condition_type_id"],
            max_reservoir_update_values=max_reservoir_update_values,
        )
    return condition_groups


def _build_condition_group_from_components(
    *,
    name: str,
    component_groups: list[TensorStatsGroup],
    reservoir_size: int,
    seed: int,
    max_reservoir_update_values: int,
) -> TensorStatsGroup:
    group = TensorStatsGroup(
        name,
        reservoir_size=reservoir_size,
        seed=seed,
        max_reservoir_update_values=max_reservoir_update_values,
    )
    if not component_groups:
        return group

    sample_count = int(component_groups[0].sample_count)
    channel_offset = 0
    height_width = _extract_height_width(component_groups[0])
    for component_group in component_groups:
        group.global_stats.merge(component_group.global_stats)
        for channel_index in sorted(component_group.per_channel):
            new_index = channel_offset + channel_index
            group.per_channel[new_index] = copy.deepcopy(component_group.per_channel[channel_index])
            group.channel_sample_counts[new_index] = int(component_group.channel_sample_counts[channel_index])
        channel_offset += len(component_group.per_channel)

    group.sample_count = sample_count
    if height_width is not None:
        group.shape_distribution[_shape_key((channel_offset, height_width[0], height_width[1]))] = sample_count
    group.channel_count_distribution[channel_offset] = sample_count
    return group


def _extract_height_width(group: TensorStatsGroup) -> tuple[int, int] | None:
    if not group.shape_distribution:
        return None
    shape_key = next(iter(group.shape_distribution))
    shape = tuple(int(value) for value in shape_key.split("x"))
    if len(shape) != 3:
        return None
    return int(shape[1]), int(shape[2])


def _condition_mapping_from_preset(preset_name: str) -> dict[str, dict[str, Any]]:
    if preset_name != "pairwise_non_target_modalities":
        raise ValueError(f"Unsupported condition preset: {preset_name}")
    mapping = {
        "rgb_plus_normal": {
            "components": ["model_rgb", "model_normal"],
            "note": "Matches the observed real type-0 training tensors in rect_cond_real_npys.",
            "is_inferred": False,
        },
        "rgb_plus_edge_depth": {
            "components": ["model_rgb", "edge_depth"],
            "note": "Inferred preset for full-data stats; not explicitly defined anywhere in the repo.",
            "is_inferred": True,
        },
        "normal_plus_edge_depth": {
            "components": ["model_normal", "edge_depth"],
            "note": "Inferred preset for full-data stats; not explicitly defined anywhere in the repo.",
            "is_inferred": True,
        },
    }
    return {
        spec["label"]: {
            **mapping[spec["label"]],
            **spec["flags"],
            "label": spec["label"],
            "legacy_condition_type_id": spec["legacy_condition_type_id"],
        }
        for spec in CONDITION_VARIANTS
    }


def _load_equirectangular_payload(sample_path: Path) -> dict[str, Any]:
    from edge3d.tensor_format import load_sample_modalities

    payload = load_sample_modalities(sample_path, decode_model_normal=True)
    payload["edge_depth"] = np.nan_to_num(payload["edge_depth"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    payload["model_depth"] = np.nan_to_num(payload["model_depth"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    payload["model_rgb"] = np.asarray(payload["model_rgb"], dtype=np.float32)
    payload["model_normal"] = np.asarray(payload["model_normal"], dtype=np.float32)
    return payload


def _derive_training_views_from_payload(
    payload: dict[str, Any],
    mapping: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    height = int(payload["model_depth"].shape[1])
    width = int(payload["model_depth"].shape[2])
    conditions_by_label: dict[str, np.ndarray] = {}
    for variant in CONDITION_VARIANTS:
        label = variant["label"]
        spec = mapping[label]
        components = [_flatten_equirectangular_component(payload, component_name) for component_name in spec["components"]]
        conditions_by_label[label] = np.concatenate(components, axis=0).astype(np.float32, copy=False)

    return {
        "input": payload["edge_depth"][:1].astype(np.float32, copy=False),
        "target": payload["model_depth"][:1].reshape(1, height, width).astype(np.float32, copy=False),
        "conditions_by_label": conditions_by_label,
    }


def _process_equirectangular_samples(
    *,
    sample_paths: list[Path],
    mapping: dict[str, dict[str, Any]],
    reservoir_size: int,
    max_reservoir_update_values: int,
    seed: int,
    include_raw_modalities: bool,
    num_workers: int,
    progress_every: int,
) -> tuple[dict[str, TensorStatsGroup], dict[str, TensorStatsGroup], dict[str, TensorStatsGroup]]:
    if num_workers <= 1 or len(sample_paths) <= 1:
        return _process_equirectangular_chunk(
            sample_paths=[str(path) for path in sample_paths],
            mapping=mapping,
            reservoir_size=reservoir_size,
            max_reservoir_update_values=max_reservoir_update_values,
            seed=seed,
            include_raw_modalities=include_raw_modalities,
            progress_every=progress_every,
        )[:3]

    groups, _ = _create_primary_groups(
        reservoir_size=reservoir_size,
        seed=seed,
        max_reservoir_update_values=max_reservoir_update_values,
    )
    condition_component_groups = _create_condition_component_groups(
        reservoir_size=reservoir_size,
        seed=seed,
        max_reservoir_update_values=max_reservoir_update_values,
    )
    extra_groups = _create_extra_groups(
        reservoir_size=reservoir_size,
        seed=seed,
        max_reservoir_update_values=max_reservoir_update_values,
        include_raw_modalities=include_raw_modalities,
    )
    num_tasks = min(len(sample_paths), max(num_workers, num_workers * 4))
    chunks = [chunk for chunk in np.array_split(np.array(sample_paths, dtype=object), num_tasks) if len(chunk) > 0]
    processed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _process_equirectangular_chunk,
                sample_paths=[str(path) for path in chunk.tolist()],
                mapping=mapping,
                reservoir_size=reservoir_size,
                max_reservoir_update_values=max_reservoir_update_values,
                seed=seed + worker_index * 100000,
                include_raw_modalities=include_raw_modalities,
                progress_every=0,
            )
            for worker_index, chunk in enumerate(chunks)
        ]
        for task_index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            worker_groups, worker_component_groups, worker_extra_groups, worker_processed = future.result()
            groups["input"].merge(worker_groups["input"])
            groups["target"].merge(worker_groups["target"])
            for component_name, worker_component_group in worker_component_groups.items():
                condition_component_groups[component_name].merge(worker_component_group)
            for group_name, worker_group in worker_extra_groups.items():
                extra_groups[group_name].merge(worker_group)
            processed += int(worker_processed)
            print(f"Merged task {task_index}/{len(futures)} ({processed}/{len(sample_paths)} NPZ samples)")
    return groups, condition_component_groups, extra_groups


def _process_equirectangular_chunk(
    *,
    sample_paths: list[str],
    mapping: dict[str, dict[str, Any]],
    reservoir_size: int,
    max_reservoir_update_values: int,
    seed: int,
    include_raw_modalities: bool,
    progress_every: int,
) -> tuple[dict[str, TensorStatsGroup], dict[str, TensorStatsGroup], dict[str, TensorStatsGroup], int]:
    groups, _ = _create_primary_groups(
        reservoir_size=reservoir_size,
        seed=seed,
        max_reservoir_update_values=max_reservoir_update_values,
    )
    condition_component_groups = _create_condition_component_groups(
        reservoir_size=reservoir_size,
        seed=seed,
        max_reservoir_update_values=max_reservoir_update_values,
    )
    extra_groups = _create_extra_groups(
        reservoir_size=reservoir_size,
        seed=seed,
        max_reservoir_update_values=max_reservoir_update_values,
        include_raw_modalities=include_raw_modalities,
    )
    for index, sample_path_str in enumerate(sample_paths, start=1):
        payload = _load_equirectangular_payload(Path(sample_path_str))
        groups["input"].update(payload["edge_depth"][:1])
        groups["target"].update(payload["model_depth"][:1])
        condition_component_groups["model_rgb"].update(_flatten_equirectangular_component(payload, "model_rgb"))
        condition_component_groups["model_normal"].update(_flatten_equirectangular_component(payload, "model_normal"))
        condition_component_groups["edge_depth"].update(_flatten_equirectangular_component(payload, "edge_depth"))
        if extra_groups:
            extra_groups["raw_model_depth_all_hits"].update(payload["model_depth"])
        if progress_every > 0 and (index % progress_every == 0 or index == len(sample_paths)):
            print(f"Processed {index}/{len(sample_paths)} NPZ samples")
    return groups, condition_component_groups, extra_groups, len(sample_paths)


def _flatten_equirectangular_component(payload: dict[str, Any], component_name: str) -> np.ndarray:
    if component_name == "model_rgb":
        tensor = np.asarray(payload["model_rgb"], dtype=np.float32)
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3])
    if component_name == "model_normal":
        tensor = np.asarray(payload["model_normal"], dtype=np.float32)
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3])
    if component_name == "model_depth":
        return np.asarray(payload["model_depth"], dtype=np.float32)
    if component_name == "edge_depth":
        return np.asarray(payload["edge_depth"], dtype=np.float32)
    raise ValueError(f"Unsupported equirectangular component: {component_name}")


def _write_group_outputs(
    *,
    output_dir: Path,
    groups: dict[str, TensorStatsGroup],
    condition_groups: dict[str, TensorStatsGroup],
    extra_groups: dict[str, TensorStatsGroup],
    save_hist: bool,
    hist_bins: int,
) -> dict[str, str]:
    generated_files = {
        "summary_json": str(output_dir / "summary.json"),
        "input_per_channel_csv": str(output_dir / "input_per_channel.csv"),
        "target_per_channel_csv": str(output_dir / "target_per_channel.csv"),
        "condition_all_per_channel_csv": str(output_dir / "condition_all_per_channel.csv"),
        **{
            f"condition_{spec['label']}_per_channel_csv": str(output_dir / f"condition_{spec['label']}_per_channel.csv")
            for spec in CONDITION_VARIANTS
        },
    }

    _write_channel_csv(output_dir / "input_per_channel.csv", groups["input"].channel_rows())
    _write_channel_csv(output_dir / "target_per_channel.csv", groups["target"].channel_rows())
    _write_channel_csv(output_dir / "condition_all_per_channel.csv", groups["condition_all"].channel_rows())
    for spec in CONDITION_VARIANTS:
        label = spec["label"]
        _write_channel_csv(
            output_dir / f"condition_{label}_per_channel.csv",
            condition_groups[label].channel_rows(),
        )

    for group_name, group in extra_groups.items():
        csv_name = f"{group_name}_per_channel.csv"
        generated_files[csv_name] = str(output_dir / csv_name)
        _write_channel_csv(output_dir / csv_name, group.channel_rows())

    if save_hist:
        _save_histogram(output_dir / "input_hist.png", "Input values", groups["input"].global_stats.reservoir_values(), hist_bins)
        _save_histogram(output_dir / "target_hist.png", "Target values", groups["target"].global_stats.reservoir_values(), hist_bins)
        _save_histogram(
            output_dir / "condition_all_hist.png",
            "Condition values (all types)",
            groups["condition_all"].global_stats.reservoir_values(),
            hist_bins,
        )
        generated_files["input_hist_png"] = str(output_dir / "input_hist.png")
        generated_files["target_hist_png"] = str(output_dir / "target_hist.png")
        generated_files["condition_all_hist_png"] = str(output_dir / "condition_all_hist.png")
        for spec in CONDITION_VARIANTS:
            label = spec["label"]
            hist_path = output_dir / f"condition_{label}_hist.png"
            _save_histogram(
                hist_path,
                f"Condition values ({label})",
                condition_groups[label].global_stats.reservoir_values(),
                hist_bins,
            )
            generated_files[f"condition_{label}_hist_png"] = str(hist_path)
        for group_name, group in extra_groups.items():
            hist_path = output_dir / f"{group_name}_hist.png"
            _save_histogram(hist_path, group_name, group.global_stats.reservoir_values(), hist_bins)
            generated_files[f"{group_name}_hist_png"] = str(hist_path)

    return generated_files


def _resolve_data_config(config: dict[str, Any], split: str, config_path: Path) -> tuple[dict[str, Any], str]:
    if "data" in config:
        if split not in config["data"] or config["data"][split] is None:
            raise ValueError(f"Experiment config does not define data.{split}")
        data_cfg = copy.deepcopy(config["data"][split])
        experiment_name = str(config.get("experiment_name") or config_path.stem)
    else:
        data_cfg = copy.deepcopy(config)
        experiment_name = config_path.stem
    data_cfg["manifest_path"] = str(_resolve_config_path_value(data_cfg["manifest_path"], base_dir=config_path.parent))
    if data_cfg.get("root_dir") is not None:
        data_cfg["root_dir"] = str(_resolve_config_path_value(data_cfg["root_dir"], base_dir=config_path.parent))
    data_cfg["horizontal_flip_p"] = 0.0
    return data_cfg, experiment_name


def _resolve_output_dir(output_dir: str | None, experiment_name: str) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return (REPO_ROOT / "analysis" / "tensor_stats" / experiment_name).resolve()


def _resolve_config_path_value(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    direct_candidate = (base_dir / path).resolve()
    if direct_candidate.exists():
        return direct_candidate
    for ancestor in [base_dir, *base_dir.parents]:
        candidate = (ancestor / path).resolve()
        if candidate.exists():
            return candidate
    return direct_candidate


def _ensure_chw_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError(f"Expected a CHW tensor with 3 dimensions, got shape {tuple(array.shape)}")
    return np.asarray(array, dtype=np.float32)


def _shape_key(shape: tuple[int, ...]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _compute_percentiles(values: np.ndarray, q: tuple[int, ...]) -> np.ndarray:
    try:
        return np.percentile(values, q, method="linear")
    except TypeError:
        return np.percentile(values, q, interpolation="linear")


def _counter_to_json_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_channel_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in CSV_COLUMNS})


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _save_histogram(path: Path, title: str, values: np.ndarray, bins: int) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    if values.size == 0:
        axis.text(0.5, 0.5, "No finite values", ha="center", va="center")
        axis.set_axis_off()
    else:
        axis.hist(values, bins=bins)
        axis.set_xlabel("Value")
        axis.set_ylabel("Count")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _print_console_summary(
    groups: dict[str, TensorStatsGroup],
    condition_groups: dict[str, TensorStatsGroup],
) -> None:
    print("Summary:")
    for display_name, group in (
        ("input", groups["input"]),
        ("target", groups["target"]),
        ("condition_all", groups["condition_all"]),
    ):
        _print_group_summary(display_name, group)
    for spec in CONDITION_VARIANTS:
        label = spec["label"]
        _print_group_summary(f"condition_{label}", condition_groups[label])


def _print_group_summary(name: str, group: TensorStatsGroup) -> None:
    global_summary = group.global_stats.summary()
    print(
        "  "
        f"{name}: samples={group.sample_count}, "
        f"min={_format_stat(global_summary['min'])}, "
        f"max={_format_stat(global_summary['max'])}, "
        f"p01={_format_stat(global_summary['p01'])}, "
        f"p99={_format_stat(global_summary['p99'])}, "
        f"mean={_format_stat(global_summary['mean'])}, "
        f"std={_format_stat(global_summary['std'])}"
    )


def _format_stat(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


if __name__ == "__main__":
    main()