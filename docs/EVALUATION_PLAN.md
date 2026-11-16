# Evaluation plan for preference disambiguation in tourism content

The primary question is whether separating viewing interest from stated travel preference improves recommendations when the two targets conflict, and whether decision-valued matched clarification achieves that improvement with fewer requested judgments. This is a technical evaluation with an exploratory human validation component. It does not measure bookings, actual visits, emissions, or a causal change in tourist behavior. Acceptance by a journal cannot be guaranteed by a protocol or an implementation.

## Evidence and claims

Use three clearly separated evidence layers:

1. Controlled simulation on source-derived tourism cards: aligned, mixed, conflicting, and deliberately nonlinear users. The last group violates the linear utility and response-scale assumptions. Simulated feedback is generated from an independent truth draw, not from the inference particles. The response-generating parameters stay fixed during HPO.
2. A real, source-grounded catalog: SynthTRIPs knowledge-base snapshot, immutable revision and SHA-256 recorded. At development time the adapter returned 200 European cities and 400 text cards. These are derived editorial cards, not observed social-media posts, users, or behavioral interactions. Two cards differ in presentation order, not video quality. Source budget and walkability categories are coarse dataset attributes, not independently verified current travel facts. Unknown critical values fail planning admission. For real deployment, `require_verified=true` requires independent verification and will reject this unaudited catalog.
3. Human stated-preference evidence: target 36 consenting adult participants, with 24 as an exploratory fallback if recruitment is slow. Each participant receives 12 cards, six viewing warmups, 24 matched query-pool pairs, 12 disjoint held-out matched pairs and 24 independent item ratings. Six warmups plus 48 pool answers plus 24 held-out answers plus 24 ratings = 102 responses, with ties/skips in pair judgments. Expectation of 15–25 minutes is a planning estimate to test with 3–5 pilot participants. Do not claim an adequate sample from that target alone; use the power sensitivity script and report uncertainty.

The human instrument collects all pool responses before algorithmic replay. It validates target separation, held-out prediction, and retrospective policy comparison. It does not establish that an adaptive conversation reduces experienced burden: query count in replay is a counterfactual algorithmic cost. Ratings and pair judgments share a participant/session and are not independent observations. Randomized pair orientation and target order are logged, but asking both targets can anchor responses. Disclose this limitation. A live randomized study can be future work; do not invent it for the deadline.

Simulation is not a substitute for human evidence for the behavioral distinction. If no human data can be collected within ten days, present a narrower proof of concept with a mechanism evaluation and avoid claiming established real-user effectiveness. JITT permits technical-theoretical and proof-of-concept work, but requires a substantive tourism/IT contribution. This design is an informed proposal, not an editorial minimum or promise of acceptance.

## Research questions

- RQ1: Does the dual utility model improve held-out travel judgments and list utility when media and travel preferences conflict?
- RQ2: Does finite-action EVSI provide better utility per requested judgment than random, entropy, disagreement-only or UCB elicitation?
- RQ3: Which components matter: target separation, aspect transfer, uncertainty, query cost, diversity and posterior rejuvenation?
- RQ4: Does the controller remain valid under missing critical attributes, response skipping, response noise and nonlinear utilities?
- RQ5: Are computational cost, evidence validity and latency compatible with an interactive tourism recommender?

## 4.1 Hardware and Software Environment

Use one attached RTX PRO 6000 Blackwell for the frozen local 7B LLM, with BF16 inference and no fine-tuning. CPU FP64 NumPy/SciPy implements SMC and finite-action EVSI. GPU acceleration of this small numerical workload is not assumed to be beneficial. Log actual Python, NumPy, SciPy, PyTorch, CUDA, driver and GPU information in every result; do not hard-code an environment that was not observed. Install a Blackwell-compatible PyTorch build in MoLab and verify CUDA before model inference. The code records the model's immutable Hugging Face revision. Save `pip freeze` with the run.

Official MoLab documentation currently describes 4 default CPU cores, 32 GB RAM, a 96 GB GPU, sessions up to 12 hours and shutdown after 90 minutes idle. Those are service limits, not guaranteed throughput. Ordinary generated files are not automatically durable. Use a nine-hour soft run budget; export results and feature caches before stopping. Existing completed user/method episodes are resumed, while a single interrupted episode is rerun deterministically.

Table 4.1: observed environment, precision, model/revision and software versions. Include GPU extraction time separately from CPU decision latency.

## 4.2 Datasets

The implemented primary adapter is SynthTRIPs, not a fake amalgamation of unrelated datasets. Freeze the raw CSV commit and checksum; deterministic cards retain exact text provenance and row references. Use six shared aspects (nature, heritage, food, nightlife, art, shopping), one known presentation variable and six aspect-availability masks, and budget/walkability compatibility plus their availability masks. The feature schema and scaling are fixed before evaluation.

Candidate catalog split: destination-hash 60/20/20 into train/validation/test. All cards for one destination stay in one split. The optional empirical-Bayes objective of manuscript Eq. 17 is not used: proper fixed priors suffice. Reserve train for development/schema choices, validation for HPO, and test for the frozen run. User identities and seeds used for HPO are distinct from test identities/seeds. Within each episode, six viewing warmup pairs, 12 held-out pairs and the query pool do not overlap. Test pair responses and item relevance labels never enter action construction, query selection, model updates or prompts.

Human evaluation is separate from simulation; do not pool their sample sizes. Survey participants see a source-derived 12-city card catalog and confirm the trip scenario and objective priorities. Their warmup/query/test pair roles are fixed in the signed-off protocol. Independent participant-level ratings define an empirical stated-utility reference; its optimum is not psychological ground truth.

TravelPlanner can be a supplementary constraint stress test, but it is not used as if it contained paired media/travel labels. It would require a separate task adapter and is not necessary for the ten-day core experiment. Likewise hotel reviews and POI check-ins do not supply the missing dual-target labels. Adding more public dataset names would not resolve this validity issue.

Table 4.2: raw and admitted cities/cards, missing budget/walkability rates, aspect coverage, source split counts, participant count, pool/test judgments, tie and skip rates. Provide catalog attrition counts, not just the raw 200 cities.

## 4.3 Evaluation metrics

Predeclare final net list utility as the primary simulation endpoint. Primary human endpoint: held-out travel pair NLL, supported by empirical rating-based net utility and viewing/travel NDCG. The exporter selects net utility for simulation and travel NLL for human evidence, with separate Holm families. Human inference remains exploratory unless the frozen sample-size and quality criteria are met.

- Net utility = reference list utility minus a FIXED evaluation cost (0.002) per requested target judgment. Controller cost is a separate HPO parameter. A matched question costs two judgments, even when skipped. Report both queries and judgments. Objective weights are (0.25,0.60,0.15) in simulation and participant-confirmed in human replay.
- Reference regret = best reference utility over the SAME represented action set minus selected-list reference utility. It is not regret over the full catalog. Human references are normalized 0–4 stated ratings. Numerical SMC posterior regret is diagnostic and separate from reference regret.
- NDCG@5 for the actual returned ordered list under each target, using continuous nonnegative relevance (simulated sigmoid utility or human rating/4) as gain. Also run K=3 and K=10 for larger catalogs; human K=5 remains the primary fixed setting. These gains do not create unobserved interaction labels.
- Held-out ordered-logistic NLL, three-class Brier score and accuracy, separately for V/T, excluding skips with denominators reported. A Brier score alone is not a proof of calibration. Score RMSE is a diagnostic against known simulation scores or normalized stated ratings, not calibrated booking probabilities.
- Pairwise taxonomy diversity, duplicated destinations, constraint violations and missing-critical rejection. Zero violations are a validity check shared by all grounded methods, not evidence of superiority.
- Requested/answered judgments, stopping point, observed skips and utility–budget curves. No claims about measured satisfaction or conversation burden from batch replay.
- Cold/warm extraction time, per-query EVSI latency p50/p95, full episode decision time, ESS/resamples/MH acceptance, approximate action gap on tiny exhaustive catalogs, peak GPU memory and cache size.

Never use Recall/MRR as the primary outcome without a defensible relevant-item definition. Do not threshold implicit interest into actual travel. Do not report transfer coefficient recovery as human psychological identification: g and b are not uniquely identifiable.

Use 3 test seeds (11/29/47), 30 independently generated users per regime per seed: 360 simulated users per method across four regimes. This is a bounded target, adjustable only before inspecting test outcomes following measured pilot costs. Share catalog, truth draws, response realizations, initial particles and action pool across methods. Average repeated inference seeds within each HUMAN participant before any statistical test. Simulated users are distinct by seed and user ID. Resample users, not lists, turns or answer rows. Report paired differences with 95% user bootstrap intervals and sign-flip p values; apply Holm correction across a predeclared primary comparison family. Secondary metrics are descriptive unless another family is declared in advance. The included report exports both predeclared endpoint families, matched-user counts, held-out answer/skip denominators and a run-completion audit. The standalone human NLL script is an optional cross-check.

Table 4.3: endpoint definitions, reference labels, aggregation unit and direction. Human and simulation tables remain separate.

## 4.4 Hyperparameter Optimization

Eight bounded candidates: particles {256,512}, controller per-judgment cost {0.001,0.004}, response-model scale {0.7,1.0}. Use validation cities and four validation users per regime at seed 7; select mean net utility evaluated with fixed burden 0.002. Priors N(0,1), Beta(2,2), tie margin 0.2, ESS threshold M/2 and two symmetric Gaussian random-walk MH moves are fixed. This is a transparent small validation search, not a comprehensive optimizer search. Do not select a different response-generating simulator per candidate. All simulated response noise is held fixed.

Primary list cap K=5, 24 candidates, 64 deterministic represented actions, 48 pool pairs, 12 shortlisted pairs, maximum five clarification queries, skip forecast 0.15. Actual feasible candidate counts can be below 24 after attrition and split. Action construction is a seeded catalog-only bounded projection/MMR search; it is shared by all methods and frozen for hypothetical answers. Its bound and approximation must be disclosed. Learned action generation or complete catalog optimization is not claimed.

Run sensitivities on validation first, then predeclare selected test sensitivities: particle count 128/256/512/1024; shortlist 4/12/all; action cap 16/64/128; skip response rates 0/0.15/0.4; response noise 0.5/1/1.5; context budget caps 1/2/3; and priorities (0.7,0.2,0.1), (0.25,0.6,0.15), (0.2,0.3,0.5). Use a reduced fixed set of 10 users per regime for secondary sweeps, three seeds, never another full tuning grid on test data. `scripts/sensitivity.py` generates and runs these declared one-factor changes.

## 4.5 Overall Performance

Show V/T metrics together so improved travel fit cannot conceal worse viewing utility. Plot net utility and reference regret against maximum query budget 0–5. Carry forward early-stopped outcomes when forming curves; do not drop stopped sessions. Stratify aligned/mixed/conflicting/nonlinear users and report actual observed pair disagreement. Include a matched human comparison table and report confidence intervals wide enough to reflect recruitment size.

Expected direction is a hypothesis, not an outcome: full transfer need not win for aligned users, independent utilities can be competitive after abundant travel evidence, and EVSI can stop too early if action coverage or particle support is poor. The code has no requirement that the proposed method outperform baselines.

## 4.6 Ablation Study

| Variant | Mechanism removed | Expected diagnostic |
|---|---|---|
| single_utility | V/T target distinction | Performance under conflict |
| shared_transfer | Travel-specific shared-aspect coefficients (g=1) | Harm from unrestricted transfer |
| independent | Aspect transfer (g=0) | Cold-start benefit/cost of sharing |
| no_questions | All clarification | Value of additional evidence |
| travel_only | Viewing half of matched comparison | Cost of missing matched evidence |
| disagreement_matched | Decision-value screen | Uncertainty versus useful information |
| no_cost | Interaction-cost stopping | Requested judgments versus small gains |
| no_rejuvenation | Posterior-invariant MH move | ESS alone versus support impoverishment |
| no_diversity | Diversity objective | Relevance/variety tradeoff |

Travel-only and matched queries request different numbers of judgments. Present per-query AND per-judgment curves and use `scripts/judgment_budget.py` to compare under the same judgment cap. Do not claim that same turn count alone establishes lower interaction cost. A same-pair versus different-pair matchedness intervention would need a separate acquisition design; the current travel-only test isolates the additional target, not every possible effect of matching.

## 4.7 Complexity Analysis

Let N be admitted items, D all utility dimensions, M particles, A represented lists, K list size, Q shortlisted pairs, O<=16 answer outcomes and H accepted observations. Utility calculation is O(MND), list matrix construction O(MAK), and EVSI O(QMAO). Query ranking uses no LLM calls once features are cached. Memory includes O(MD+MN+MA+N²+AK), plus the stored history. MH cost is O(RMHD) per rejuvenation round for pair observations; the implementation currently recomputes historical item utilities in its likelihood loop, so actual cost can be O(RMHND) and must be profiled rather than quoting the optimized bound as measured complexity. Catalog diversity stores an N² distance matrix.

Bounded action construction performs up to 5A trials, each greedily selecting K destinations; its deterministic heuristic can omit a good action. `scripts/action_gap.py` compares the represented Bayes action with all ordered lists on a tiny catalog. This is an approximation check, not a full-catalog guarantee.

`tourism benchmark` profiles 12 size settings and multiple repetitions. Report query-only timings separately from process startup, LLM download/extraction, catalog loading and disk writes. Measure peak CUDA memory for extraction using MoLab's GPU and save it with the environment. Development CPU timings cannot be presented as RTX results.

## 4.8 Baseline Comparison

| Implementation | Evidence and acquisition | Role |
|---|---|---|
| popularity | Grounded source popularity, no extra questions | Nonpersonalized reference, not actual engagement frequency |
| bt_no_questions | Independent MAP ordered-logistic content models, warmup only | Conventional content/preference baseline |
| bt_random | Same MAP models with random matched observations | Point-estimation versus Bayesian transfer |
| random_matched | Same full posterior, random matched pairs | Acquisition control |
| entropy_matched | Same posterior, maximum predictive outcome entropy | Uncertainty-based acquisition |
| disagreement_matched | Same posterior, target-reversal probability | Isolate disagreement heuristic |
| ucb_matched | Same posterior, absolute mean plus SD of pair difference | Transparent BO-inspired heuristic |
| llm_direct (optional GPU run) | Frozen Qwen list selector with random_matched's EXACT history and action lists | Direct LLM ranking on equal evidence |

These are explicit baselines and controlled adaptations. `ucb_matched` is NOT PEBOL: PEBOL includes an NLI-based item-belief update and language acquisition. `llm_direct` is NOT an official reproduction of LLM4Rerank, TRACE, GATE or DiMA. Cite those works for positioning only unless their official full implementations are actually run on a justified task adapter. A strong recent external comparison remains desirable; budget the direct frozen-LLM run to provide a different model family without falsely renaming a heuristic. The direct model is evaluated on utility/regret and generation validity; it does not expose probabilities for the pairwise NLL table. Invalid outputs are retained, reported and assigned zero reference utility in the conservative all-attempt analysis.

Also compare lexical and LLM feature backends with identical evidence splits. The LLM is a frozen Qwen2.5-7B-Instruct with immutable revision, greedy decoding and 384 output-token cap. Exact-span checks ensure source linkage but NOT semantic correctness. Export a blinded feature audit with `scripts/audit.py`; two annotators can judge semantic support on a sampled set of records, resolve disagreements and report raw agreement/precision. Absence of a recognized aspect is missingness, not a negative preference label. An LLM judge alone is insufficient as ground truth for the central behavioral claim.

## 4.9 Discussions

Discuss when transfer helps and harms, matched response anchoring, skip forecast misspecification, posterior nonidentifiability, finite action/query coverage, source metadata missingness and staleness, editorial text-card versus actual social-feed validity, European city coverage, small convenience samples, within-session test-item exposure, and retrospective versus live conversation differences. Discuss feasibility without turning list diversity into a sustainability claim. Provide trace examples of supported feature evidence, rejected unsupported spans, excluded critical-unknown items, zero-query decisions and conflicting judgments. Do not cherry-pick only successful examples.

## Ten-day execution plan

| Day | Deliverable |
|---|---|
| 1 | Freeze protocol and catalog, measure extraction/inference speed, test instrument on 3–5 pilot participants |
| 2 | Extract/cache LLM features, audit semantic support, finalize prompts/schema and validation-only HPO |
| 3–5 | Collect human judgments in parallel with bounded core simulation; export checkpoints every session |
| 6 | Run missingness/noise/weight sensitivities and numerical convergence/action-gap checks |
| 7 | Run direct LLM baseline, targeted feature ablation and latency profiles |
| 8 | Freeze data exclusions and aggregate paired user results, intervals and corrected tests |
| 9 | Write Chapter 4, revise overclaims in Chapters 1–3, prepare figures and evidence appendix |
| 10 | Reproduce selected runs from clean outputs, check references and anonymization, submit |

Planning budget: reserve one GPU session for extraction and one for direct ranking, and multiple CPU sessions for the main numerical experiments. These are allocations, not measured runtime promises. Use actual pilot seconds to estimate remaining workload before launching a full suite. If the budget is exceeded, cut secondary sensitivity combinations first and record the change before examining test outcomes. Preserve the four simulator regimes, the core controls, three seeds and the human evidence wherever possible.

## Sources checked

- Journal scope: https://link.springer.com/journal/40558/aims-and-scope
- Author guidelines, quantitative methods and code/data availability: https://link.springer.com/journal/40558/submission-guidelines
- Official MoLab limits/storage: https://docs.marimo.io/guides/molab/
- SynthTRIPs dataset and code: https://huggingface.co/datasets/ashmib/SynthTRIPs and https://github.com/ashmibanerjee/synthTRIPS-code
- TravelPlanner official task: https://github.com/OSU-NLP-Group/TravelPlanner
- PEBOL primary paper: https://arxiv.org/abs/2405.00981
- Qwen model card: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

Rechecked journal and MoLab guidance on 16 November 2026. JITT explicitly asks for experimental conditions/manipulation checks, questionnaire items and sample-size adequacy; code/data access is strongly encouraged and a data-availability statement is required. Its citation style is author–year with an alphabetized reference list; the uploaded numeric references need conversion. It uses double-anonymous review, so an author-identifying GitHub URL must not be carelessly placed in reviewer-facing materials. Use an anonymized archive when appropriate. No human recruitment, consent collection, GPU run or journal submission is implied by the availability of this code.

## Runtime calibration command

Before the main experiment, run `python scripts/estimate_runtime.py --catalog data/catalog_llm.json --config results/hpo/selected.json`. The saved plan times every numerical method on validation conflict users and scales to 5,760 episodes, with a twofold planning allowance. It excludes LLM inference, HPO, recruitment, reporting and downloads; repeat on the actual MoLab session. `bash scripts/pipeline.sh backup` exports both results and feature caches.
