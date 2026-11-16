# Running the evaluation on MoLab

Use the attached GPU for frozen LLM extraction and direct ranking. Numerical inference is a small CPU FP64 workload; leaving the GPU active does not accelerate NumPy automatically. First measure a short run, then allocate the remaining sessions. No official daily quota or uninterrupted availability is assumed.

## Set up one session

Attach the GPU in the notebook specifications. In the terminal, clone the repository and install:

```bash
git clone https://github.com/sepehrmustafavi/Tourism.git
cd Tourism
python -m pip install -e '.[llm,plots]'
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m unittest discover -s tests -v
python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
python -m pip freeze > environment-freeze.txt
```

If using a downloaded source ZIP, unpack it instead of cloning. If the notebook has no terminal, run equivalent commands in a Python cell with `subprocess.run([sys.executable, '-m', ...], check=True, cwd=repo_path)`. Do not paste notebook magic or IPython syntax into a marimo Python cell.

A CUDA-enabled PyTorch version must support the actual Blackwell GPU. Installation of the `llm` extra alone does not prove GPU compatibility. Fix any CUDA import/kernel error before extraction. The 7B BF16 model needs a download and substantial GPU memory, but no training. The service's 96 GB allocation is ample in principle; record measured memory and actual offloading.

## Validate the data and model path

```bash
python -m tourism.cli fetch --out data/catalog_raw.json
python -m tourism.cli extract --catalog data/catalog_raw.json --out data/catalog_lexical.json --backend lexical
python -m tourism.cli run --catalog data/catalog_lexical.json --config configs/smoke.json --out results/local_smoke --users 2 --seeds 11
python -m tourism.cli extract --catalog data/catalog_raw.json --out data/catalog_llm.json --backend llm
python scripts/audit.py --catalog data/catalog_llm.json --out data/semantic_audit.csv --records 100
```

LLM extraction is cached per source text, source version, prompt, model revision and backend. Interrupted extraction reuses successful cached records. Invalid JSON is logged and produces no accepted feature; it is not secretly replaced with lexical output. Inspect rejection rates and semantic support before freezing the representation. Copy the audit CSV for independent annotators. Never change the aspect mapping after examining test results.

## Freeze and run

```bash
python -m tourism.cli tune --catalog data/catalog_llm.json --config configs/main.json --out results/hpo --users 4
python -m tourism.cli run --catalog data/catalog_llm.json --config results/hpo/selected.json --out results/main --users 30 --seeds 11 29 47 --max-seconds 32400
python -m tourism.cli report --runs results/main --out results/report
python scripts/plot.py --summary results/report/summary.json --out results/figures
```

A nine-hour soft cap leaves room before the documented 12-hour service limit. If paused, export results immediately. In the next session restore the same source, data snapshot, cached features, selected config and result files, then rerun the identical command. Completed user/method records are skipped; one interrupted record restarts. The signature prevents mixing different code/data/config. An ordinary ZIP is not automatically durable on MoLab: download it to your machine, or explicitly upload it through the MoLab sidebar / supported persistent storage route.

```bash
python scripts/archive.py --source results --out tourism_results.zip
# Back up the data/ directory, including raw snapshot, model-feature output and cache, separately.
python scripts/archive.py --source data --out tourism_data.zip
```

Keep source code, feature snapshots, environment-freeze.txt and both ZIPs. Do not commit raw human responses to a public repository. For a future session you can upload your own trusted ZIP through the sidebar and restore it into `results/` or `data/` using `shutil.unpack_archive`. Ordinary files written by Python alone are not promised to persist; the official MoLab documentation currently says persistence applies to sidebar uploads or persistent-cache mechanisms subject to limits.

## Secondary experiments and human validation

Follow the commands in README for the same-judgment-budget comparison, one-factor sensitivities, missingness, tiny exhaustive search diagnostic, latency benchmark, human instrument/replay and direct LLM selector. The direct selector is a separate GPU phase and emits its own all-attempt report. Freeze the human exclusion/quality rules before aggregation, average inference seeds within each participant, and analyze held-out travel NLL as the human primary endpoint.

Do not launch all optional sweeps immediately. Use the saved sensitivity manifest to budget a bounded secondary run without modifying it based on favorable test outcomes. Keep environment and both cold/warm timing measurements with the run. MoLab storage and compute policies may change: https://docs.marimo.io/guides/molab/.
