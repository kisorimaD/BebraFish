#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/activate.sh"
cd "$TASK_ROOT/src"
exec "$TASK_ROOT/env/bin/python" -m training.plane20.train_center_mask \
    --data "$TASK_ROOT/dataset" --output "$TASK_ROOT/runs/center_mask32" "$@"
