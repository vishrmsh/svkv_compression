# SVKV compression decision pilot

A standalone experiment testing whether support-vector data description can
select useful KV cache entries for a frozen language model. Attention over the
retained entries remains ordinary softmax. `plain_jane` is a read-only reference;
this project neither imports it nor needs it to run.

**Decision: do not commit to a full paper on the current SVDD ranker.**
The completed pilot contains 54 contexts and 1,350 answers. At 20% retained KV,
mean custom-needle recall is 31.9% for post-RoPE SVDD and 29.2% for pre-RoPE
SVDD, versus 100% for KeyDiff and leverage scoring. Small natural-QA results
are mixed. This rejects the tested implementation as a promising starting
point; it does not rule out every SVDD design.

Start with the [decision memo](docs/decision_memo.md), then the
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
.venv/bin/python scripts/benchmark_systems.py --method full
.venv/bin/python scripts/benchmark_systems.py --method svdd --fraction .2
.venv/bin/python scripts/benchmark_systems.py --method keydiff --fraction .2
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
.venv/bin/python scripts/run_systems_suite.py
.venv/bin/python scripts/summarize_systems.py
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
| `references.bib` | Primary-source bibliography |

The `.gitignore` excludes the environment, downloaded weights, original dataset
and full tokenized cases. Large repetitive solver logs have deterministic gzip
copies; the uncompressed copies are excluded from Git. The manifests and experiment evidence are
intended to remain with the repository.

## Scientific boundaries

SVDD here is a blockwise, budget-truncated dual-coefficient ranker. It does not
inherit the older coefficient-gated memory's exact deletion theorem. A small
`nu` does not guarantee a small support set. The benchmark tests post-prefill,
question-hidden compression, not bounded-memory online prefill. The original
full cache stays alive in the paired quality harness; use the isolated systems
measurements for process-memory interpretation. Baseline scorers are documented
adaptations rather than claims to reproduce every original serving system.

See [third-party sources and assets](THIRD_PARTY_NOTICES.md). Model weights and
benchmark source data remain subject to their original terms.
