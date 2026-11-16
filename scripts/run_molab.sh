#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Use a workspace path you can export; ordinary local files are NOT durable on MoLab.
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -m tourism.cli run --catalog data/catalog_llm.json --config results/hpo/selected.json --out results/main --users 30 --max-seconds 32400
"$PYTHON_BIN" -m tourism.cli report --runs results/main --out results/report
"$PYTHON_BIN" scripts/archive.py --source results --out tourism_results.zip
