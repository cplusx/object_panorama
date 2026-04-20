#!/usr/bin/env bash
set -euo pipefail

python -u -m edge3d.generation.export_training_tensors_parallel \
  --dataset-root /home/devdata/edge3d_data \
  --output-dir /home/devdata/edge3d_data/equirectangular_data \
  --cache-dir /tmp/edge3d_objaverse_cache \
  --start-idx 0 \
  --end-idx 30000 \
  --gpu-ids 0,1 \
  --workers-per-gpu 6 \
  --summary-every 50 \
  --download-timeout-sec 60