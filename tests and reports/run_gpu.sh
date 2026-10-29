#!/usr/bin/env bash
set -euo pipefail
benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$benchmark_dir/.."
# Activate the GPU machine's environment before running this script.
# Additional CLI flags may be supplied as arguments.
python -m gnn_benchmark.cli run --device cuda "$@"
