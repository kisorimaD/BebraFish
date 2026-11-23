#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source training/plane20/activate.sh
exec python -m training.plane20.run_ablation "$@"
