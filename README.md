# SVKV compression

An empirical evaluation of support-vector data description (SVDD) for KV-cache
selection in a frozen language model. Selected K/V tensors are physically
compacted, and attention over the retained entries uses ordinary softmax.

The main experiment contains 54 contexts and 1,350 answers. At 20% retained KV,
mean custom-needle recall is 31.9% for post-RoPE SVDD and 29.2% for pre-RoPE
SVDD, versus 100% for KeyDiff and leverage scoring. Small natural-QA results
are mixed. The tested SVDD rankers do not improve the quality–cost tradeoff
over the geometric baselines in this evaluation.

A separate decision-score evaluation yields **45.8%**
needle recall at 20% KV versus **100%** for KeyDiff and leverage (360 separate
new answers). Repeating the original selection probes with exact linear and
same-RBF kernel leverage leaves a **2.5-point synthetic advantage** over kernel
leverage, but **no ICU advantage**. See the [additional comparisons](docs/followup_findings.md)
for paired intervals and the limits of that small synthetic result.

Start with the [results and analysis](docs/results.md), then the
[literature review](docs/literature_review.md) and
[prior-work audit](docs/prior_work_status.md). The
[pilot protocol](docs/pilot_protocol.md) and
[selector definitions](docs/selector_method.md) state the exact scope.

The evaluated model is Qwen2.5-7B-Instruct with four-bit weights and 16-bit KV
cache on an Apple M3 Ultra. The pilot has 54 contexts, 25 arms per context:
full cache and eight selectors at 10%, 20% and 50% retention. It includes
36 custom NIAH cases at 8k/16k/32k windows plus 18 untruncated examples across
three LongBench tasks. Custom NIAH is not an official RULER evaluation.

## Reproduce locally

Use Python 3.11 on an Apple Silicon Mac with a working Metal GPU. The current
runtime is validated only for the included Qwen2/Qwen2.5 architecture. Model
and data downloads are about 4.4 GB and stay in this directory.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python scripts/download_assets.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python -m pytest -q
SVKV_RUN_METAL_TESTS=1 .venv/bin/python -m pytest -q tests/test_runtime.py tests/test_systems_cache.py
.venv/bin/python scripts/run_pilot.py --output results/reproduction
.venv/bin/python scripts/summarize_results.py results/reproduction/predictions.jsonl
.venv/bin/python scripts/validate_artifacts.py --results results/pilot
```

The validation command audits the delivered main-run evidence and archived
execution source; use its default before making changes. The dependency lock
records the Apple environment actually used; it is not a
portable CUDA requirements file. CPU selector tests can be run separately with
NumPy, SciPy, scikit-learn and pytest. A restricted process sandbox may hide
Metal; run GPU commands from a normal local terminal if needed.
Use a new output directory after changing code or inputs; the pilot's resume
mechanism skips completed arms and is not a substitute for source-version checks.

For one short pipeline check:

```bash
.venv/bin/python scripts/run_pilot.py --limit 1 --output results/my_smoke
```

For an isolated memory and fixed-compute timing measurement:

```bash
.venv/bin/python scripts/benchmark_systems.py --method full --output-directory results/my_systems
.venv/bin/python scripts/benchmark_systems.py --method svdd --fraction .2 --output-directory results/my_systems
.venv/bin/python scripts/benchmark_systems.py --method keydiff --fraction .2 --output-directory results/my_systems
```

Run systems commands sequentially with no other accelerator job. Each invocation
loads a fresh model process and frees the full cache before compressed decode.
All arms use reserved append buffers in these systems runs, avoiding the
quality harness's repeated concatenation costs. Logical payload and allocated
capacity are both recorded.
Weights are hashed by default; `--skip-weight-hash` avoids repeated hashing
after checking `data/asset_manifest.json`.

To repeat the complete serial systems suite and summarize it after the quality
job has finished:

```bash
.venv/bin/python scripts/run_systems_suite.py --output-directory results/systems_reproduction
.venv/bin/python scripts/summarize_systems.py results/systems_reproduction --output results/systems_reproduction_summary
.venv/bin/python scripts/package_evidence.py
```

## Evidence and source map

| Location | Contents |
|---|---|
| `src/svkv/` | Selectors, physical cache runtime, inverse-RoPE geometry, tasks/metrics |
| `scripts/` | Asset download, data preparation, pilot, summary, isolated systems runs |
| `configs/pilot_protocol.json` | Fixed method/data settings and preflight addition record |
| `data/case_manifest.json` | Exact sample IDs, lengths, seeds and token hashes |
| `data/asset_manifest.json` | Model/tokenizer/archive file hashes |
| `results/pilot/predictions.jsonl` | Every main-run answer, score, timing and byte count |
| `results/pilot/summary/` | Per-task and per-length CSV tables, paired differences, PNG/PDF figures |
| `results/pilot/validation.json` | Complete evidence audit: 54 cases, 1,350 answers, exact bytes and score checks |
| `results/pilot/solver_diagnostics.jsonl.gz` | Every SVDD block's support/convergence record (raw copy also retained locally) |
| `results/pilot/execution_code/` | Snapshot of the code used for the main run |
| `results/pilot_run.log` | Main-run progress and raw output excerpts |
| `results/systems/` | Fresh-process memory and fixed-decode measurements |
| `results/systems_summary/` | Isolated systems medians and observed ranges |
| `results/runtime_validation.xml` | Real-model cache and geometry checks |
| `results/systems_cache_validation.xml` | Reserved append-buffer equivalence checks |
| `results/final_validation.xml` | Final combined validation: all 79 tests passed, including real-model Metal checks |
| `configs/decision_followup.json` | Frozen 36-case decision-score protocol, no parameter sweep |
| `results/decision_followup/` | 360 separate answers, archived source, compressed solver records, and validated summary/figures |
| `configs/redundancy_check.json` | Original selection protocols plus fixed exact linear and kernel leverage |
| `results/redundancy/` | Synthetic trial evidence, aggregate-only ICU results, paired intervals, source snapshots, and figure |
| `results/followup_validation.xml` | Current suite: 102 passed, including real-model checks; 2 optional CVXPY tests skipped |
| `results/redundancy_validation.xml` | CPU selection-probe suite: all 10 passed, including the optional CVXPY checks |
| `references.bib` | Primary-source bibliography |

The `.gitignore` excludes the environment, downloaded weights, original dataset
and full tokenized cases. Large repetitive solver logs have deterministic gzip
copies; the uncompressed copies are excluded from Git. The manifests and experiment evidence are
intended to remain with the repository.

## Reproduce the additional experiments

After preparing the same pinned model and cases as above, use a new output
directory for the fixed decision-score rerun:

```bash
.venv/bin/python scripts/run_decision_followup.py --output results/decision_reproduction
.venv/bin/python scripts/summarize_decision_followup.py --results results/decision_reproduction
```

The summary script's defaults validate the delivered follow-up. These cases
were already inspected in the pilot; the follow-up is not a held-out test.
Both alpha and decision ranking select globally across fitted blocks, without
equal per-block quotas. The original alpha comparator is read from the original
pilot and is not pooled with new answers.

The original selection probes run on CPU in a separate environment:

```bash
python3.11 -m venv .venv-redundancy
.venv-redundancy/bin/python -m pip install -r requirements-redundancy-lock.txt
.venv-redundancy/bin/python -m pip install --no-deps -e .
.venv-redundancy/bin/python -m pytest -q tests/test_redundancy.py
.venv-redundancy/bin/python scripts/run_redundancy_check.py --task synthetic --output results/synthetic_reproduction
```

To reproduce ICU as well, supply the original trusted, credentialed local
cache with `--clinical-cache /path/to/icu_vitals_n1500.npz` and omit
`--task synthetic`. Clinical data are not distributed. Only cohort-level
clinical summaries are written. The [protocol](docs/redundancy_protocol.md)
specifies the preserved preprocessing, event channel, budgets, and readouts.
The aggregate figure can be regenerated with
`.venv/bin/python scripts/plot_redundancy.py results/redundancy`.
Audit the public synthetic and aggregate clinical evidence with
`.venv-redundancy/bin/python scripts/validate_redundancy_results.py`;
this audit does not access the clinical cache.

## Scientific boundaries

SVDD in the KV experiments uses blockwise fits, with either dual-coefficient
ranking or the fixed decision-score follow-up. Neither inherits
the older coefficient-gated memory's exact deletion theorem. A small
`nu` does not guarantee a small support set. The benchmark tests post-prefill,
question-hidden compression, not bounded-memory online prefill. The original
full cache stays alive in the paired quality harness; use the isolated systems
measurements for process-memory interpretation. Baseline scorers are documented
adaptations rather than claims to reproduce every original serving system.

See [third-party sources and assets](THIRD_PARTY_NOTICES.md). Model weights and
benchmark source data remain subject to their original terms.
