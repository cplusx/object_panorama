#!/usr/bin/env bash
set -euo pipefail

# python tools/overfit_rectangular_conditional_jit.py \
#   configs/experiment/edge3d_flow_overfit.yaml \
#   --device cuda \
#   --num-samples 4 \
#   --max-steps 3000

train_pid=""

cleanup_train_group() {
  local initial_signal="${1:-TERM}"
  if [[ -n "${train_pid}" ]]; then
    local train_pgid="${train_pid}"
    kill -"${initial_signal}" -- "-${train_pgid}" 2>/dev/null || true
    wait "${train_pid}" 2>/dev/null || true
    kill -KILL -- "-${train_pgid}" 2>/dev/null || true
    train_pid=""
  fi
}

handle_signal() {
  local signum="$1"
  trap - EXIT INT TERM
  if [[ "$signum" -eq 2 ]]; then
    cleanup_train_group INT
  else
    cleanup_train_group TERM
  fi
  exit "$((128 + signum))"
}

handle_exit() {
  local exit_code="$1"
  trap - EXIT INT TERM
  cleanup_train_group TERM
  exit "$exit_code"
}

trap 'handle_signal 2' INT
trap 'handle_signal 15' TERM

CUDA_VISIBLE_DEVICES=0,1 \
setsid python tools/train_lightning_rectangular_conditional_jit.py \
  configs/experiment/edge3d_flow_train.yaml \
  --device cuda \
  --precision 32-true \
  -r &
train_pid=$!

trap 'handle_exit $?' EXIT
wait "${train_pid}"