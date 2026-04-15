import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Edge3D equirectangular export under multiple GPU/worker configurations.")
    parser.add_argument("--dataset-root", default="/home/devdata/edge3d_data")
    parser.add_argument("--profile-root", default="/home/devdata/edge3d_data/equirectangular_profile_runs")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--uids-json", default=None)
    return parser.parse_args()


def _run(command: list[str], cwd: Path) -> float:
    start = time.perf_counter()
    subprocess.run(command, cwd=str(cwd), check=True)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    dataset_root = Path(args.dataset_root)
    profile_root = Path(args.profile_root)
    profile_root.mkdir(parents=True, exist_ok=True)

    if args.uids_json is None:
        from edge3d_pipeline import ObjaverseEdgeDataset

        dataset = ObjaverseEdgeDataset(dataset_root)
        selected = dataset.available_ids[: args.sample_count]
        uids_json = profile_root / "profile_uids.json"
        uids_json.write_text(json.dumps({"uids": selected}, indent=2), encoding="utf-8")
    else:
        uids_json = Path(args.uids_json)

    runs = [
        {
            "name": "1gpu_1worker",
            "command": [
                sys.executable,
                "edge3d_export_training_tensors_parallel.py",
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(profile_root / "out_1gpu_1worker"),
                "--cache-dir",
                str(profile_root / "cache_1gpu_1worker"),
                "--uids-json",
                str(uids_json),
                "--gpu-ids",
                "0",
                "--workers-per-gpu",
                "1",
                "--summary-every",
                "10",
            ],
        },
        {
            "name": "2gpu_1worker_each",
            "command": [
                sys.executable,
                "edge3d_export_training_tensors_parallel.py",
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(profile_root / "out_2gpu_1worker_each"),
                "--cache-dir",
                str(profile_root / "cache_2gpu_1worker_each"),
                "--uids-json",
                str(uids_json),
                "--gpu-ids",
                "0,1",
                "--workers-per-gpu",
                "1",
                "--summary-every",
                "10",
            ],
        },
        {
            "name": "2gpu_2worker_each",
            "command": [
                sys.executable,
                "edge3d_export_training_tensors_parallel.py",
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(profile_root / "out_2gpu_2worker_each"),
                "--cache-dir",
                str(profile_root / "cache_2gpu_2worker_each"),
                "--uids-json",
                str(uids_json),
                "--gpu-ids",
                "0,1",
                "--workers-per-gpu",
                "2",
                "--summary-every",
                "10",
            ],
        },
    ]

    results = []
    for run in runs:
        out_dir = Path(run["command"][run["command"].index("--output-dir") + 1])
        cache_dir = Path(run["command"][run["command"].index("--cache-dir") + 1])
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        elapsed = _run(run["command"], cwd=repo_root)
        summary = json.loads((out_dir / "export_summary.json").read_text(encoding="utf-8"))
        success_count = int(summary.get("count", 0))
        failed_count = int(summary.get("failed_count", 0))
        requested = len(json.loads(uids_json.read_text(encoding="utf-8"))["uids"])
        results.append(
            {
                "name": run["name"],
                "elapsed_sec": elapsed,
                "requested_count": requested,
                "success_count": success_count,
                "failed_count": failed_count,
                "sec_per_requested": elapsed / max(requested, 1),
                "sec_per_success": elapsed / max(success_count, 1),
                "throughput_success_per_sec": success_count / max(elapsed, 1e-8),
                "summary_path": str(out_dir / "export_summary.json"),
            }
        )

    baseline = next(item for item in results if item["name"] == "1gpu_1worker")
    for item in results:
        item["speedup_vs_1gpu_1worker"] = baseline["elapsed_sec"] / max(item["elapsed_sec"], 1e-8)

    report = {
        "sample_count": len(json.loads(uids_json.read_text(encoding="utf-8"))["uids"]),
        "uids_json": str(uids_json),
        "results": results,
    }
    out_path = profile_root / "profile_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()