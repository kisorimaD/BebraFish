#!/usr/bin/env bash
set -e

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/activate.sh"
cd "$script_dir/../.."
export OPENBLAS_NUM_THREADS="${PLANE20_DENOISER_BLAS_THREADS:-1}"
exec python -m training.plane20.train_denoiser \
    --data "$TASK_ROOT/dataset_ablation" \
    --detectors "$TASK_ROOT/runs/ablation_center32" \
    --output "$TASK_ROOT/runs/denoiser_normal" "$@"
