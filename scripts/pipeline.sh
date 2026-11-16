#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:-verify}"
case "$STAGE" in
verify)
  "$PYTHON_BIN" -m unittest discover -s tests -v
  "$PYTHON_BIN" -m tourism.cli fixture --out data/fixture.json --items 200
  "$PYTHON_BIN" -m tourism.cli run --catalog data/fixture.json --config configs/smoke.json --out results/smoke --users 1 --seeds 11
  "$PYTHON_BIN" -m tourism.cli report --runs results/smoke --out results/smoke_report
  ;;
prepare)
  "$PYTHON_BIN" -m tourism.cli fetch --out data/catalog_raw.json --revision 12a333da6c9036ac49444d9199cb690470b443ac
  "$PYTHON_BIN" -m tourism.cli extract --catalog data/catalog_raw.json --out data/catalog_llm.json --backend llm
  "$PYTHON_BIN" scripts/audit.py --catalog data/catalog_llm.json --out data/semantic_audit.csv --records 100
  ;;
tune)
  "$PYTHON_BIN" -m tourism.cli tune --catalog data/catalog_llm.json --config configs/main.json --out results/hpo --users 4
  "$PYTHON_BIN" scripts/estimate_runtime.py --catalog data/catalog_llm.json --config results/hpo/selected.json
  ;;
main) bash scripts/run_molab.sh ;;
report)
  "$PYTHON_BIN" -m tourism.cli report --runs results/main --out results/report
  "$PYTHON_BIN" scripts/plot.py --summary results/report/summary.json --out results/figures
  ;;
backup)
  "$PYTHON_BIN" scripts/archive.py --source results --out tourism_results.zip
  "$PYTHON_BIN" scripts/archive.py --source data --out tourism_data.zip
  ;;
*) echo 'Usage: bash scripts/pipeline.sh {verify|prepare|tune|main|report|backup}' >&2; exit 2 ;;
esac
