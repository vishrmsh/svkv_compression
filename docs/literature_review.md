# SV selection for KV compression: initial literature review

Research cut-off and access date: **2026-09-27**. This is a targeted, primary-source review for a go/no-go pilot, not an exhaustive survey or a priority certificate. Publication dates below are initial arXiv dates unless stated otherwise. The accompanying [`references.bib`](../references.bib) contains the cited papers.

## Decision-relevant finding

**Query-agnostic selection of geometrically unusual keys is already an established KV-compression direction.** The closest comparisons are **KeyDiff**, **Compactor's leverage-score component**, and **K-norm**, not only SnapKV and StreamingLLM. A support-vector objective could still be a distinct algorithm, but the broad story “preserve rare information through key geometry without seeing the question” is occupied. KeyDiff explicitly motivates its selector through distinctive keys; Compactor explicitly motivates outlier retention through rare future queries. [KeyDiff](https://arxiv.org/html/2504.15364v4), [Compactor](https://arxiv.org/html/2507.08143v2), [K-norm](https://arxiv.org/abs/2406.11430).

The targeted searches below did **not find a separate paper explicitly applying support-vector data description (SVDD) to transformer KV eviction**. That is evidence about this search, not proof of novelty. The likely defensible contribution would be a scalable, budget-controlled SVDD selector with a demonstrated advantage over those close geometric baselines. A new solver name alone is unlikely to justify a full paper.

There is also a necessary correction to the proposed theory narrative: **zero SV coefficient does not imply output-neutral eviction in ordinary softmax attention.** The existing guarantee is about a coefficient-gated readout. Selection-only softmax discards that gate and therefore does not inherit the guarantee. The pilot should investigate empirical quality, without advertising certified lossless eviction.

## Relationship to the existing preprints

SV Attention is currently arXiv **v4, revised September 11, 2026**, with the user's stated title. Its abstract carefully limits exact removal to the current normalized coefficient-weighted readout, and separates that from future admissions and audited decremental updates. This scope is compatible with leaving it as a preprint. [SV Attention v4](https://arxiv.org/abs/2607.12204v4).

GemmaSV is also **v4, revised September 11, 2026**. It concerns auditable removal of addressable conversation rows inside frozen Gemma 3. Its current abstract reports limited admission, residual distinguishability from never-stored memory, no update-speed advantage in the FP32 proxy, and weak retained-answer matching. These results motivate a separate compression experiment; they are not evidence of competitive KV compression. [GemmaSV v4](https://arxiv.org/abs/2607.27539v4).

The rare-group and ICU selection results are promising hypotheses about minority coverage. A selector that preserves statistical outliers can nevertheless discard ordinary but answer-critical text. The new task must test actual frozen-model outputs at matched cache budgets, including distractors that are more geometrically unusual than the answer.

## Closest prior work

| Work | Signal and scope | Implication for SV selection |
|---|---|---|
| **K-norm**, June 2024 | Retains small-L2-norm keys, based on an observed inverse relationship with attention importance; no attention matrix needed. | Extremely cheap key-only baseline. “Large norm means important” is not the paper's rule. [Paper](https://arxiv.org/abs/2406.11430) |
| **KeyDiff**, April 2025; NeurIPS 2025; v4 January 2026 | Retains keys dissimilar to an average key direction; includes blockwise prefill and key-diversity theory. | Closest direct rival to a query-free boundary/outlier selector. Compare both quality and scoring time. [Paper](https://arxiv.org/abs/2504.15364v4) |
| **LagKV**, April 2025 | Uses lag-relative comparisons among cached K/V states without attention weights. | Another cheap query-free redundancy signal; useful second-stage baseline. [Paper](https://arxiv.org/abs/2504.04704) |
| **KVzip**, May 2025 | Builds a cache reusable across unseen questions by scoring importance during context reconstruction. Extra model computation is part of the method. | Query-agnostic does not mean key-only. Compare amortized construction cost if reuse is the intended benefit. [Paper](https://arxiv.org/abs/2505.23416) |
| **Compactor**, July 2025, v2 December 2025 | Combines approximate leverage scores on pre-position-encoded keys with noncausal context-attention scores. Includes context-specific budget calibration and systems work. | Strongest overlap with the rare-information motivation. Leverage-only is an ablation, not the full Compactor method. [Paper](https://arxiv.org/html/2507.08143v2) |
| **CurDKV**, September 2025 | Value-guided selection through approximate CUR decomposition/leverage scores. | Key coverage alone does not guarantee preservation of the attention output; values matter. [Paper](https://arxiv.org/abs/2509.15038) |
| **Expected Attention**, October 2025 | Estimates future-query attention from activation distributions and ranks KV contributions. | Strong no-future-question comparator; uses more model information than key-only SVDD. [Paper](https://arxiv.org/abs/2510.00636) |
| **WildCat**, February 2026 | Selects a weighted coreset with randomly pivoted Cholesky and reconstruction weighting; includes KV-cache experiments and bounded-input approximation theory. | A close kernel/coreset reference if the project pivots toward approximation guarantees. Its weighted attention is distinct from unweighted subset softmax. [Paper](https://arxiv.org/abs/2602.10056) |
| **CapKV**, April 2026 | Information-capacity/log-determinant objective under a linear-Gaussian attention surrogate; leverage-based selection. | Geometric objectives and formal surrogate theory are already crowded. State exactly which objective any guarantee preserves. [Paper](https://arxiv.org/abs/2604.25975) |
| **CoRDS**, May 2026 | Joint-KV geometric coreset coverage and diversity for streaming video models. | Adjacent modality, but blocks any broad claim to first geometric/coreset KV selection. [Paper](https://arxiv.org/abs/2605.14310) |
| **TaskPress**, August 2026 | Task-guided cache construction before downstream questions, plus outlier information from quantization scales. | Task-conditional reuse is another occupied framing; distinguish strictly key-only from task-guided query-agnostic selection. [Paper](https://arxiv.org/abs/2608.03276) |

### Cheap baseline formulas and implementation provenance

For each head, let keys have shape `N × D`, and retain the highest scores under an identical token budget.

**K-norm:** use `score_i = -||k_i||_2` (or an inverse-norm score with the same ranking). This favors **low** key norms. Ordinary orthogonal RoPE preserves each key's L2 norm, though architecture-specific scaling can require care. [K-norm paper](https://aclanthology.org/2024.emnlp-main.1027.pdf).

**KeyDiff in the currently inspected KVPress source:** normalize individual keys, average their directions into an anchor, and use negative cosine similarity to that anchor:

```text
anchor = mean_i(k_i / max(||k_i||, eps))
score_i = -cosine_similarity(k_i, anchor)
```

The KeyDiff paper's efficient experimental variant averages raw keys before the cosine computation; KVPress currently averages normalized keys. Record the variant. One-shot scoring also differs from the paper's blockwise eviction schedule. [KVPress KeyDiff source](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/keydiff_press.py).

Version check: **v4, January 20, 2026**, retains these scoring and anchor choices in Section 3.2. Its additions include phonebook lookup, correlation analysis, mobile-device scoring latency, and discussion of the community KVPress RULER benchmark. The added evidence strengthens its relevance; it does not change the baseline formula used here. [KeyDiff v4](https://arxiv.org/html/2504.15364v4).

**Leverage-only in the inspected KVPress source:** recover pre-RoPE keys, center them along the sequence, Gaussian-project from `D` to `r=48`, and score

```text
X = (K - mean_sequence(K)) @ Phi       # Phi_ij ~ Normal(0, 1/r)
G = X.T @ X + 0.01 I
score_i = max(0, x_i.T @ solve(G, x_i))
```

Use a Cholesky solve rather than an explicit inverse. The source adds adaptive jitter on failure and standardizes scores; standardization does not change a head's ranking. Its centering, Cholesky, and jitter choices must be disclosed when reproducing it. This algorithm avoids an `N × N` matrix. [KVPress leverage source](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/leverage_press.py).

**Position handling is an actual ablation.** Applying key geometry after RoPE mixes content geometry with position-dependent rotations. Leverage scoring in Compactor uses pre-RoPE keys. Compare SVDD pre-RoPE versus post-RoPE on a development set, then freeze the choice. Do not silently compare an unfavorable representation of a baseline against a tuned representation of SVDD. [Compactor method](https://arxiv.org/html/2507.08143v2).

## Established baselines requested for the project

| Method | What it retains | Evaluation qualification |
|---|---|---|
| **SnapKV** | Per-head important positions inferred from attention in a trailing observation window, with clustering/pooling. | A question in that window provides extra information. Use both clearly labeled question-visible and context-only protocols in a full study. [Paper](https://arxiv.org/abs/2404.14469) |
| **H2O** | A mix of recent entries and accumulated attention heavy hitters. | Online accumulated state and recency allocation are part of the baseline, not just a one-time top-k. [Paper](https://arxiv.org/abs/2306.14048) |
| **PyramidKV** | Attention-based retention with unequal layer budgets. | Match total cache bytes across layers; equal per-layer counts do not reproduce its allocation. [Paper](https://arxiv.org/abs/2406.02069) |
| **StreamingLLM** | Initial attention sinks plus a recent window. | Designed for stable streaming behavior, not retrieval of arbitrary evicted middle content. An initial-plus-recent one-shot baseline should be labeled as an adaptation if position treatment differs. [Paper](https://arxiv.org/abs/2309.17453) |
| **TOVA** | Eviction driven by the current/last token's attention, aggregated over heads. | The temporal eviction schedule and aggregation rule matter. A prefill-only approximation is not the original online experiment. [Paper](https://arxiv.org/abs/2401.06104) |
| **Keyformer** | Attention-derived scoring with stochastic/Gumbel-based treatment to identify influential tokens. | Despite its name, this is not a selector based solely on key-vector geometry. [Paper](https://arxiv.org/abs/2403.09054) |
| **Random / uniform-stride / initial-plus-recent** | No learned importance signal. | Essential controls. Test whether a more expensive geometric score actually beats simple coverage and recency. |

KVPress provides many relevant scorers and benchmark infrastructure, but adapter names alone do not establish reproduction of original papers. Its README explicitly separates context and question, supports quantized caches, and warns that its `BlockPress` is not a true chunked-prefill implementation. Pin the library commit and distinguish a scorer comparison from an end-to-end online policy reproduction. [KVPress repository](https://github.com/NVIDIA/kvpress).

## Quantization changes the meaningful memory comparison

**KIVI** uses asymmetric low-bit treatment of keys and values; **KVQuant** studies low-bit KV representation including pre-RoPE and outlier handling. **TurboQuant** adds random-rotation quantization with explicit distortion analysis. These retain information at every position while reducing bytes per element, unlike eviction. [KIVI](https://arxiv.org/abs/2402.02750), [KVQuant](https://arxiv.org/abs/2401.18079), [TurboQuant](https://arxiv.org/abs/2504.19874).

A nominal 20% FP16 cache uses approximately the same payload bytes as a full cache at 3.2 bits per element, before metadata, scales, alignment, residual windows, and selector state. Therefore “fivefold token reduction” alone is not a sufficient systems result. Compare **actual allocated KV bytes**, peak process/device memory, construction time, and generation latency. A full-cache quantization baseline and an eviction-plus-quantization combination should be included before a paper claim. A weight-quantized model with an FP16 KV cache is not a KV-quantization experiment.

Do not transfer reported speedups between papers: model size, batch size, kernel, hardware, retained precision, and whether prefill is counted vary. A local Apple/MLX pilot can compare its own methods fairly; it does not establish a CUDA serving speedup.

## What the old theorem does and does not buy

The SV readout has coefficients in its definition:

```math
o_\alpha(q)=\frac{\sum_i \alpha_i\kappa(q,k_i)v_i}{\sum_i\alpha_i\kappa(q,k_i)}.
```

If `alpha_j=0`, deleting that term changes neither sum, with the other fitted state fixed. This is the current-state removal statement in SV Attention. [SV Attention v4](https://arxiv.org/abs/2607.12204v4).

Ordinary softmax over a retained subset has no `alpha` factor:

```math
o_S(q)=\frac{\sum_{i\in S}\exp(q^Tk_i/\sqrt d)v_i}{\sum_{i\in S}\exp(q^Tk_i/\sqrt d)}.
```

**Counterexample derived for this review.** Take scalar keys `(-1,0,1)` and SVDD kernel `exp(-(x-y)^2/4)`, with `nu=0.5`, hence cap `C=2/3`. The optimum is `alpha=(1/2,0,1/2)`: both active coordinates have `(K alpha)_i=(1+exp(-1))/2≈0.68394`, while the inactive coordinate has `(K alpha)_2=exp(-1/4)≈0.77880`, satisfying the convex program's KKT conditions. Now set the softmax query to zero and values to `(0,1,0)`. Full attention returns `1/3`; keeping only SV support returns `0`. The violation occurs immediately, with no new token and no refit.

A valid elementary softmax statement instead depends on **discarded attention mass**. Let `m_D(q)` be the full-softmax probability on discarded entries. For a nonempty retained set,

```math
o(q)-o_S(q)=m_D(q)(o_D(q)-o_S(q)).
```

If `||v_i||≤M`, this gives `||o-o_S||≤2M m_D(q)`. This is an independent algebraic observation, not a new SV-specific theorem. An SV certificate would need to control that mass over a stated query class, and then handle propagation through later layers and generated tokens. The recent minimax analysis in **The risk of KV cache compression** is relevant to any such claim. [Paper](https://arxiv.org/abs/2607.01520).

### Sparsity, budgeting, and scalable fitting

With `sum(alpha)=1` and `alpha_i≤1/(nu N)`, elementary arithmetic gives `|support(alpha)|≥nu N`. It gives **no upper bound**. Small `nu` permits a sparse solution but does not force one. For a narrow RBF kernel, `K` approaches the identity and the optimum approaches uniform coefficients on all keys. Selecting the largest `B` coefficients or distances is a budgeted heuristic; it may discard positive coefficients and must be described as such. Boundary distance and dual coefficient ranking are also different methods.

At `N=32,768`, one FP32 Gram matrix has `N²` entries and occupies **4 GiB per head**, before solver workspaces. Blocks of `b` keys reduce the materialized Gram size to `b²`; independently fitting each block does not solve the full-history SVDD problem. Approximate solves need convergence, support fraction, fallback, and timing diagnostics. Coreset acceleration of SVDD itself dates to at least **Chu, Tsang and Kwok (IJCNN 2004)**, so “use a coreset to scale SVDD” is not independently new. [Author-hosted paper](https://home.cse.ust.hk/~jamesk/papers/ijcnn04.pdf), [SVDD foundation](https://research.tudelft.nl/en/publications/support-vector-data-description/).

## Evaluation needed for a defensible decision

**Separate three notions:** (1) no access to the downstream question, (2) no query vectors used by the selector, and (3) no access to future tokens during streaming prefill. A context-only SnapKV implementation satisfies the first but not the second. One-shot SV selection after a full prefill satisfies the first two but not strict streaming memory bounds.

The July 2026 **query-visibility audit** directly compares question-visible and question-hidden compression with matched budgets. It reports substantial ranking changes and makes simple baselines central. This is a warning against interpreting a question-hidden SnapKV loss as a general victory over SnapKV. [Audit](https://arxiv.org/abs/2607.11942).

For the inexpensive first stage, use an open frozen model, actual compacted K/V tensors, 8K and 32K inputs where feasible, 10/20/50% **retention** budgets, and common sinks/recent-window accounting. Include full cache, SnapKV adaptation, initial-plus-recent, random or uniform, K-norm, KeyDiff, and leverage-only. Keep the question out of compression. Use identical original positions for selected keys and correct absolute positions for appended questions/tokens. Release prompts, random seeds, raw answers, exact selected token counts, hardware/library versions, timings, and failures.

Use varied needle locations, multiple needles, plausible distractors, and aggregation/tracing, not solely a single conspicuous passkey in repetitive filler. **RULER** supplies 13 configurable tasks beyond basic needle finding; **LongBench** supplies varied natural-document workloads. A small home-generated task inspired by RULER is useful, but must not be reported as an official RULER score. [RULER](https://arxiv.org/abs/2404.06654), [LongBench](https://arxiv.org/abs/2308.14508).

The follow-on study should use official data/metrics, multiple current model families, held-out natural tasks, multi-query reuse, and at least one genuine streaming/chunked experiment. LongBench v2 adds more demanding long-document reasoning, though a small pilot model can be floor-limited on it. Report the full-cache score so compression is not judged on tasks the base model already fails. [LongBench v2](https://arxiv.org/abs/2412.15204).

Recommended decision rule, chosen before a confirmatory run: continue only if SV selection repeatedly improves the quality-versus-total-cost tradeoff against **the strongest geometric baseline**, especially at 10–20% retention; does not buy needle recall by materially worsening natural QA; and can be implemented without a quadratic full-context workspace. An inconclusive or tiny pilot merits a bounded follow-up, not a paper commitment. If the advantage disappears against KeyDiff/leverage or depends on conspicuous synthetic outliers, the broad compression-paper hypothesis is weak.

## Search log and limits

All searches and linked primary-source inspections were conducted on **2026-09-27** using a web search index, arXiv, ACL Anthology, author-hosted papers, and official source repositories. Relevant full method sections were read for KeyDiff (initial v3 inspection, then v4 verification) and Compactor v2; official KVPress scorer code was inspected for their cheap baseline variants. Abstract-level screening of the other methods is sufficient for this initial map but is not a replication audit of every paper.

| Search family | Exact representative queries | Outcome |
|---|---|---|
| Direct SVDD prior art | `"support vector data description" "KV" cache`; `"support vector" "KV cache"`; `"KV cache" "SVDD"`; `"SVDD" "token eviction"`; `"KV cache" "one-class" selection`; `"Support Vector" "key-value cache"` | No separate transformer SVDD-eviction paper found. Broader strings produce anomaly detection, singing-voice deepfake detection (another SVDD acronym), and storage-cache SVM results; these are not matching methods. |
| Geometry and novelty | `"KV cache" "query agnostic" compression`; `"KV cache" geometric coreset key norm compression`; `"KeyDiff" "KV"`; `"KV cache" "coreset" arxiv` | KeyDiff, Compactor, KVzip, LagKV, WildCat, CoRDS, TaskPress, and the query-visibility audit. Followed paper and official-code links. |
| Named baselines | `"SnapKV" arxiv`; `"H2O" "Heavy-Hitter" arxiv`; `"PyramidKV" arxiv`; `"TOVA" "Transformers" arxiv`; `"StreamingLLM" arxiv`; `"Keyformer" arxiv`; `"A Simple and Effective L2 Norm-Based Strategy"` | Primary papers and repositories located. |
| Quantization and theory | `KIVI tuning free asymmetric 2bit quantization arxiv Liu 2024`; `"KVQuant" arxiv`; `"TurboQuant" arxiv`; `"The risk of KV cache compression"`; `"Scaling up support vector data description by using core-sets" authors` | Primary quantization papers, contemporary theory, and old SVDD coreset literature located. |
| Benchmark validity | `RULER benchmark arxiv 2404.06654`; `LongBench benchmark arxiv 2308.14508`; `"LongBench v2" arxiv` | Original benchmark definitions inspected. |

The search is limited by index coverage and terminology; unpublished work, unindexed repositories, papers using different names, and the full citation graph may be missing. Check references and forward citations of the closest methods again before submission. No claim here implies acceptance or independent reproduction of a preprint's reported performance. Links to a repository's `main` branch are observations on the access date, not immutable code citations; pin exact commits for experiments.
