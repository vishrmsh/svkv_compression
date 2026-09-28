# What the earlier projects establish

Audited 2026-09-27. This is a foundation audit for a **new KV-compression project**, not a proposed revision of either preprint. `plain_jane` was read only. No models or modules were imported from it during this audit.

## Current public status

Both public arXiv records are v4, last revised 2026-09-11:

- [SV Attention, arXiv:2607.12204v4](https://arxiv.org/abs/2607.12204v4): *What a Deletion Certificate Covers, and Where It Expires: Auditable Removal from a Support-Vector Memory*.
- [GemmaSV, arXiv:2607.27539v4](https://arxiv.org/abs/2607.27539v4): *Can an AI Assistant Really Forget? Auditable Deletion from Addressable Memory*.

The public [SV Attention repository](https://github.com/VyLabs-AI/sv-attention) and [GemmaSV repository](https://github.com/VyLabs-AI/gemmasv) now foreground auditable memory editing, numerical checks, and limitations. Some local top-level README files and historical replacement checklists retain older titles and plans. The current paper sources and recorded results are better authorities for scientific claims than those older summaries. The user intends to leave these as preprints; this audit makes no claim about a current venue decision.

## SV Attention: useful evidence, with its proper scope

The strongest transfer opportunity is **selection under redundancy**, not a demonstrated efficient LLM attention replacement.

| Existing result | What was actually measured | What it does not establish |
|---|---|---|
| Rare-group recall 0.861 versus 0.319 | Synthetic contexts with six singleton groups and six groups of ten near-duplicates; approximately 56% retained; equal-coefficient RBF readout for every selector. The comparator ranks summed similarity to evaluation queries. | Long-context LLM quality, a true online H2O implementation, or performance at 10% retention. |
| ICU event-hour retention 0.464 versus 0.225 | 718 qualifying stays; about 32% mean retention; SpO2 defines events but is withheld from every selector. Comparator is a similarity-density proxy. Recency achieved 0.409 and random 0.333. | Clinical prediction improvement or a head-to-head KV-cache baseline. |
| Clinical prediction AUROC 0.696 | Separate deterioration task; LSTM 0.783 and softmax 0.751. | Atypical-token selection is not automatically task-optimal. |
| Small language model improvement | 3.22M-parameter hybrid, first 5M bytes of enwik8, seven seeds; 2.178 versus 2.383 best-validation bits/character at matched visible-token count. | General scaling or an inference-time compression result on a pretrained model. |
| Larger fixed-budget experiments worsened | At 10M parameters, mean 2.5% worse over five 6,000-step runs; a 32M run also behind. | This is adverse evidence at those training budgets, not a proof about converged models or selection-only compression. |
| Training throughput | 9,125 tokens/s on the recorded M3 Ultra configuration, 35.8× slower than the repository's unfused softmax baseline. | Competitive system performance. |

These values and experimental qualifications were checked against local `journal_submissions/machine_learning/source/appendix_sv.tex` and `svattn_arxiv_v3/paper/data/v3_evidence.json`, beneath `/Users/vishrmsh/Documents/plain_jane`. They correspond to the bounded feasibility evidence in the [v4 paper](https://arxiv.org/html/2607.12204v4). The clinical advantage is real within its stated endpoint, but the often-used “H2O” label in older figures must not be carried into the new project as if it were the published H2O algorithm.

The latest numerical contribution is more careful than “the solver is exact.” In the sequential synthetic audit the original procedure completed 66/80 trajectories; a checking-and-rescue wrapper completed all 5,120 scheduled operations with eight initialization rescues and 84 update rescues. The largest reported readout discrepancy was 0.5273% of retained-value range. This supports a finite evaluated numerical policy, not unrestricted robustness on long model-key streams. [Current SV Attention paper](https://arxiv.org/html/2607.12204v4).

## GemmaSV: a useful integration reference, not a compression result

GemmaSV replaces the global attention operator in frozen Gemma 3. The canonical evaluated setting uses `nu=0.7`, 128-token boundaries, deterministic bandwidth sampling, and prefix-mass preservation. At 4B, all five global layers are grafted; local sliding-window layers remain native. The gate still changes attention weights, so removing that weighting in the new project is a substantive method change.

The scale check records paired PPL changes of +3.704%, +1.851%, and +11.699% at 1B, 4B, and 12B. Strict admission goes from 7/8 to 3/8, 6/8 to 6/8, and 6/8 to 5/8 respectively. Thus 4B is the strongest tested transfer point, not evidence that the same gate works across scales. At 4B the historical local certificate's maximum probe KL is 6.48e-11, but 66.4% of affected decrements need refit fallback. These local refits preserve already-contextualized surviving rows; they are distinct from rebuilding the conversation without the deleted record. [GemmaSV v4](https://arxiv.org/html/2607.27539v4).

The later matched-cost study finds no speed advantage for the current FP32 proxy: its paired update ratio is 1.012 versus graft rebuild and 35.79 versus base rebuild. This is a workload involving copied caches, masks, gates, and teacher-forced probes, not a compressed cache decoder. In particular, logical masks still retain the K/V arrays. A new compression paper must show actual tensor compaction and measured serving costs. These details were cross-checked against the public repository's [study description](https://github.com/VyLabs-AI/gemmasv) and local `gemma_sv/sv_global_attention.py`, `gemma_sv/reproducibility/README.md`, and journal study reports.

## Critical correction: the exactness theorem does not transfer to ordinary softmax

The proposed pitch currently combines two incompatible claims:

1. Keep only keys with SVDD support coefficients, then use ordinary softmax over them.
2. Evicting an SVDD zero coefficient is exactly output-neutral at eviction.

The second claim holds for the earlier **coefficient-weighted** readout. Its numerator and denominator both multiply each key's contribution by its coefficient. It does not hold for ordinary softmax, where every finite, unmasked score has strictly positive weight. This distinction is already explicitly documented in the local `svattn/kv_eviction.py` prototype.

A direct counterexample uses scalar keys `(-1, +1, 0)` and values `(0, 0, 1)`. Under a sufficiently broad RBF SVDD fit, the middle key is reserve and the endpoint coefficients are `(1/2, 1/2)`. For a dot-product query `q=0`, ordinary softmax gives all three keys weight `1/3`: the output is `1/3`. Dropping the reserve key changes it to `0`. Query independence of the selector does not change this result.

For ordinary softmax, the useful exact identity instead is:

```
o_full = (1 - delta) * o_kept + delta * o_dropped
o_full - o_kept = delta * (o_dropped - o_kept)
```

Here `delta` is the discarded **softmax attention mass for the particular query**, and the two subset outputs are separately normalized. If all values have norm at most `Vmax`, the error is at most `2 * Vmax * delta`. This is an elementary identity and bound derived here, not a new SVDD theorem. A small SVDD coefficient alone does not bound `delta`.

The defensible initial contribution is therefore: **query-independent geometric cache selection, with measured quality and cost**. The old reactivation construction motivates caution about streaming support sets. It is not itself a guarantee, or the only source of error, in the new softmax cache.

## Scaling and budget corrections

- `nu` is not the retained fraction. Since coefficients sum to one and each is at most `1/(nu*n)`, the support count is at least `ceil(nu*n)` in exact arithmetic, and may be much larger. Decreasing `nu` permits more sparsity; it does not guarantee a 10% cache. Record realized support fractions and the exact-budget rule separately.
- Keeping the largest coefficients when support exceeds a budget is a further heuristic. Capped coefficients often tie, making tie handling and ordering relevant. Do not describe top-k truncation as exact support-set pruning.
- A full 32,768-token Gram matrix has 1,073,741,824 entries: **4 GiB in fp32 per head**, or 8 GiB in fp64, before solver temporaries. Total overhead multiplies across concurrently solved heads and layers.
- A 128-token *fit interval* is not a 128-token *problem*. GemmaSV solves the whole prefix at each boundary. Its batched causal code can allocate `Kpad` with shape `(number_of_boundaries * grouped_heads, max_prefix, max_prefix)`. Reusing that code unchanged does not fix the long-context scaling problem.
- A genuinely blockwise approximation, coreset, or feature approximation changes the selection problem. Label it accordingly and compare a small instance with a full solve when assessing approximation error. Local block boundaries can bias selection; equal per-block quotas prevent adaptive allocation across blocks.
- Preserve absolute positions when compacting already-RoPE-transformed keys. Keep the native softmax scaling, query/key normalization, GQA structure, and mask semantics. Context-token budget percentages should be separated from model weights, protected recent tokens, decode growth, and temporary solver memory.

## Reusable code and pitfalls

All paths below are relative to the read-only `plain_jane` reference. Copy only needed code into this project, retain notices, and record provenance if used.

| Reference module | Reusable idea | Caution |
|---|---|---|
| `cp_svm/oneclass_qp.py` | Small independent QP reference and KKT checks | CPU solver costs; not a long-context serving algorithm. |
| `cp_svm/oneclass_fast.py` | Maintained incremental state and bordered inverse | Latest paper documents numerical failure modes and rescues. |
| `svattn/mlx_svdd.py` | Batched FISTA and capped-simplex projection | Approximate fp32 coefficients; requires dense Gram; no inherited exactness certificate. |
| `svattn/fast_diff_svdd.py` | Solver bridge and fixed-partition differentiation | Differentiation is unnecessary for a training-free selector. |
| `svattn/kv_eviction.py` | Earlier selection-only softmax prototype, including correct theory disclaimer | TinyGPT only; dense masks/full storage; catches errors and silently falls back; the `h2o` proxy sums attention over all sequence queries before prefix decisions, so it leaks future-query information. Do not use as the production baseline. |
| `experiments/p2_streaming_select.py` | Separate synthetic causal selection experiment | RBF toy readout, not an LLM KV benchmark; warmup-query comparator is a proxy. |
| `gemma_sv/sv_global_attention.py` | Gemma layer wiring, RoPE/GQA handling, cached sessions | Changes the operator, duplicates some state, and retains prefix-fit scaling costs. |
| `svattn/chunked_gate.py` | Streaming bookkeeping and explicit point-in-time scope | Its `budget` is not necessarily a hard cap when too many active points remain. |

## Local execution pointers

The read-only environment `/Users/vishrmsh/Documents/plain_jane/.venv311` exists and its interpreter resolves to Homebrew Python 3.11. A historical `.venv` also exists but points to the Xcode Python 3.9. A standalone environment in this workspace is preferable for the new project.

The Hugging Face cache at `/Users/vishrmsh/.cache/huggingface/hub` contains folders for Gemma 3 1B/4B PT and IT, Gemma 3 12B PT, Qwen3.5-4B-Base, and several MLX models. Folder presence alone does not certify a complete compatible checkpoint; the pilot should record its actual loaded revision. The Qwen3.5-4B-Base snapshot contains config/tokenizer files and two safetensors shards, but its hybrid attention architecture warrants separate compatibility review before using it as a conventional KV-cache pilot.

The prior hardware notes identify an Apple M3 Ultra and warn that concurrent MPS jobs interfere. They recommend synchronized timings, warmed runs, model revision recording, and separating smoke-run timings from research claims. The new pilot should obtain its own runtime and memory records rather than inherit old numbers.

## Decision implication before new experiments

The prior work justifies testing whether atypicality helps preserve low-frequency information at a tight cache budget. It does **not** yet justify a compression performance claim, a certified softmax-eviction claim, or an assumption of novelty. The proposed new paper should earn those claims with its own method definition, literature comparison, physical cache measurements, and long-context quality results.
