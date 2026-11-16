# Tourism preference disambiguation

A reference implementation and bounded evaluation protocol for **LLM Guided Preference Disambiguation for Multi Objective Tourism Content Recommendation**. It separates viewing interest from stated travel preference, uses aspect-specific Bayesian transfer, and requests matched comparisons only when their expected decision value exceeds the declared burden.

Start with [the manuscript review in Persian](docs/MANUSCRIPT_REVIEW_FA.md), [the complete Chapter 4 evaluation plan](docs/EVALUATION_PLAN.md), and [the MoLab execution guide](docs/MOLAB.md). No benchmark results, human study, GPU execution or journal acceptance are implied by the code. Development smoke runs are verification only.

## Ten-day execution entry point

The protocol is frozen around one source-grounded catalog, four simulated preference regimes, three seeds, 16 numerical methods/ablations, and a separate real-participant validation. Target 36 adults (24 is an exploratory fallback); do not replace them with generated respondents. Source-derived editorial text cards support a bounded tourism-content proof of concept, not a claim about real social-video behavior.

After installing the package below, execute stages separately so outputs can be exported between MoLab sessions:

```bash
bash scripts/pipeline.sh verify   # CPU fixtures; no paper results
bash scripts/pipeline.sh prepare  # pinned source; cached frozen Qwen extraction
bash scripts/pipeline.sh tune     # validation only, plus measured runtime forecast
bash scripts/pipeline.sh main     # resumable numerical suite; nine-hour soft cap
bash scripts/pipeline.sh report
bash scripts/pipeline.sh backup   # BOTH results and semantic feature caches
```

Review `results/runtime_plan.json` before the full suite. It extrapolates measured validation CPU episodes; it does not promise GPU throughput. Run the human instrument and direct-LLM comparison separately as described below. A paused run produces provisional reports with a completion audit. Re-run the identical main command to resume; changed code/data/config requires a new output directory. The automatic report uses simulation net utility and human travel NLL as separate primary endpoints and Holm families, with held-out answer/skip denominators.

## Install and verify

```bash
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The numerical core requires only NumPy/SciPy. Optional frozen local LLM inference:

```bash
python -m pip install -e '.[llm,plots]'
```

Use a PyTorch build supporting the attached GPU; verify CUDA separately. There is no fine-tuning.

## Reproducible workflow

```bash
# Resolve the dataset revision once and retain the raw catalog manifest.
python -m tourism.cli fetch --out data/catalog_raw.json
# Fast CPU control. Do not relabel lexical features as an LLM experiment.
python -m tourism.cli extract --catalog data/catalog_raw.json --out data/catalog_lexical.json --backend lexical
# GPU extraction, greedy generation, source-linked spans, immutable model revision.
python -m tourism.cli extract --catalog data/catalog_raw.json --out data/catalog_llm.json --backend llm
# Validate, freeze HPO, then use the selected config for all methods.
OPENBLAS_NUM_THREADS=1 python -m tourism.cli tune --catalog data/catalog_llm.json --config configs/main.json --out results/hpo --users 4
OPENBLAS_NUM_THREADS=1 python -m tourism.cli run --catalog data/catalog_llm.json --config results/hpo/selected.json --out results/main --users 30 --seeds 11 29 47 --max-seconds 32400
python -m tourism.cli report --runs results/main --out results/report
python scripts/plot.py --summary results/report/summary.json --out results/figures
```

The 16 numerical variants yield a target of 5,760 completed user/method episodes (4 regimes × 3 seeds × 30 users × 16 methods). An atomic JSON per episode supports resume. Rerun the identical command after restoring its output directory. A code/config/catalog signature prevents reusing stale results after changes. A partly completed episode restarts deterministically. Runtime limits are checked between episodes; they are soft limits, not mid-operation interrupts. Use separate output directories for different backends or protocols.

The source snapshot used in development contained 200 cities/400 editorial cards. Only 109 cities met the default known budget/walkability requirements; 91 had missing critical metadata. Destination-disjoint raw train/validation/test sizes were 127/35/38, with 67/24/18 admitted planning cities. Report attrition and actual per-episode candidate counts; 24 is a cap. These are source-derived text cards and coarse scenario attributes, not observed social interactions or verified current trip costs.

## Human validation

```bash
python -m tourism.cli survey --catalog data/catalog_llm.json --participant P001 --out data/surveys
# Open P001.html locally. Participant may consent, skip comparisons and download responses.
# Preserve the generated protocol file; the returned data must match it.
python -m tourism.cli replay --catalog data/catalog_llm.json --responses data/P001_responses.json --protocol data/surveys/P001_protocol.json --config results/hpo/selected.json --out results/human
python -m tourism.cli report --runs results/human --out results/human_report
python scripts/human_nll.py --users results/human_report/users.csv --out results/human_report/primary_nll.json
python scripts/power.py --n 36 --effect 0.5
```

Recruit real consenting adults yourself; the code never fabricates participants or sends invitations. No identifiers beyond pseudonyms are requested. Raw study records are ignored by git. The instrument is batch collection followed by retrospective replay, not a live adaptive trial; report its limits. Independent ratings are normalized stated-preference references, not booking outcomes. Final human NLL is the predeclared human primary endpoint. All repeated inference seeds are averaged within participant before inference. Report skip denominators and never count responses as independent participants.

## Controls and diagnostics

```bash
python scripts/sensitivity.py --catalog data/catalog_llm.json --config results/hpo/selected.json
# Review/freeze plan first; add --execute to run the one-factor sweeps.
python scripts/judgment_budget.py --catalog data/catalog_llm.json --config results/hpo/selected.json
python scripts/missingness.py --catalog data/catalog_llm.json --out data/missing30.json --rate .3
python scripts/audit.py --catalog data/catalog_llm.json --out data/semantic_audit.csv --records 100
python scripts/action_gap.py --catalog data/catalog_llm.json
python -m tourism.cli benchmark --catalog data/catalog_llm.json --config results/hpo/selected.json --out results/latency.json --repeats 10
# Optional direct frozen-LLM selector on equal histories/action lists, NOT LLM4Rerank.
python -m tourism.cli direct --catalog data/catalog_llm.json --runs results/main --out results/direct
python scripts/direct_report.py --direct results/direct --runs results/main --out results/direct_comparison.json
```

For a quick offline integrity check, use `tourism fixture` and `configs/smoke.json`. Synthetic fixtures, simulated users and human records carry separate evidence labels. The direct LLM baseline reports invalid generations and a conservative all-attempt utility analysis; it is not inserted into the probabilistic NLL table. Numerical baselines include source popularity, MAP paired-preference models, random/entropy/disagreement/UCB policies and structural ablations. UCB is explicitly a heuristic, not a PEBOL reproduction.

## Implementation map

| Manuscript | Code | Scope |
|---|---|---|
| Eq. 1 | `Catalog.feasible` | Confirmed scenario constraints; critical unknowns excluded |
| Eq. 2 | `Catalog`, `validate_features`, `extract` | Fixed aspects, missingness masks, exact source spans |
| Eqs. 3–4 | `Posterior.utilities`, `log_prior` | Dual utilities, Gaussian/Beta priors; logit Jacobian in MH |
| Eqs. 5–8 | `comparison`, `log_likelihood`, `update`, `resample_move` | Ties/skips, deduplicated events, historic contexts, SMC/MH |
| Eqs. 9–11 | `list_scores`, `build_actions` | Fixed finite action set, destination uniqueness, simplex priorities |
| Eq. 12 | `shortlist` | Posterior strict-reversal probability |
| Eqs. 13–16 | `regret`, `query_value`, `run_episode` | All <=16 joint outcomes, no hypothetical resampling, positive-value stopping |
| Eq. 17 | Fixed/validation-selected priors | Optional empirical Bayes is not used or claimed |
| Algorithms 1–2 | `run_episode`, `query_value` | Single confirmed context per episode, auditable outputs |

Question presentation uses deterministic validated templates in the experiment. Free-text interpretation and LLM paraphrasing are deliberately not evaluated. LLM extraction is the implemented semantic component; arbitrary generated text is never admitted as hard feasibility evidence. `validate_question` falls back to the canonical schema. The current experiment studies the numerical clarification mechanism, not natural-language generation quality.

## Research boundaries

Utility coefficients and transfer gates are model-dependent, not uniquely identified psychological traits. The action set and query shortlist approximate a larger task. Exact-span validation is not semantic entailment: perform the independent audit. Source records are unaudited for current real-world deployment; `require_verified=true` requires separately verified records and rejects the default source catalog. Changing context must establish a new decision problem while retaining each observation's historic context. No automatic environmental or sustainability conclusions follow from list diversity.

Review the journal's [current author guidelines](https://link.springer.com/journal/40558/submission-guidelines), provide a data-availability statement, and prepare an anonymized reviewer archive for double-anonymous review. This repository URL identifies its owner.
