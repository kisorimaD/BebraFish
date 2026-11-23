#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$TASK_ROOT/src/training/plane20/run_training.sh" \
    --output "$TASK_ROOT/runs/wide64" --base-filters 64 "$@"
