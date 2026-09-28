# Relationship to SV Attention and GemmaSV

This review summarizes the evidence and limitations of the two works that motivate the geometric cache-selection hypothesis. Public records and reported results were checked on 2026-09-27.

## Current public status

Both public arXiv records are v4, last revised 2026-09-11:

- [SV Attention, arXiv:2607.12204v4](https://arxiv.org/abs/2607.12204v4): *What a Deletion Certificate Covers, and Where It Expires: Auditable Removal from a Support-Vector Memory*.
- [GemmaSV, arXiv:2607.27539v4](https://arxiv.org/abs/2607.27539v4): *Can an AI Assistant Really Forget? Auditable Deletion from Addressable Memory*.

The [SV Attention repository](https://github.com/VyLabs-AI/sv-attention) and [GemmaSV repository](https://github.com/VyLabs-AI/gemmasv) provide implementations and evidence for auditable memory editing. The results below refer to the v4 papers.

## SV Attention: useful evidence, with its proper scope

The relevant evidence concerns **selection under redundancy**. It does not demonstrate an efficient LLM attention replacement.

| Existing result | What was actually measured | What it does not establish |
|---|---|---|
| Rare-group recall 0.861 versus 0.319 | Synthetic contexts with six singleton groups and six groups of ten near-duplicates; approximately 56% retained; equal-coefficient RBF readout for every selector. The comparator ranks summed similarity to evaluation queries. | Long-context LLM quality, a true online H2O implementation, or performance at 10% retention. |
| ICU event-hour retention 0.464 versus 0.225 | 718 qualifying stays; about 32% mean retention; SpO2 defines events but is withheld from every selector. Comparator is a similarity-density proxy. Recency achieved 0.409 and random 0.333. | Clinical prediction improvement or a head-to-head KV-cache baseline. |
| Clinical prediction AUROC 0.696 | Separate deterioration task; LSTM 0.783 and softmax 0.751. | Atypical-token selection is not automatically task-optimal. |
| Small language model improvement | 3.22M-parameter hybrid, first 5M bytes of enwik8, seven seeds; 2.178 versus 2.383 best-validation bits/character at matched visible-token count. | General scaling or an inference-time compression result on a pretrained model. |
| Larger fixed-budget experiments worsened | At 10M parameters, mean 2.5% worse over five 6,000-step runs; a 32M run also behind. | This is adverse evidence at those training budgets, not a proof about converged models or selection-only compression. |
| Training throughput | 9,125 tokens/s on the recorded M3 Ultra configuration, 35.8× slower than the repository's unfused softmax baseline. | Competitive system performance. |

These values and experimental qualifications correspond to the bounded feasibility evidence in the [v4 paper](https://arxiv.org/html/2607.12204v4). The clinical comparison concerns its stated retention endpoint. The “H2O” label used in some earlier figures denotes a similarity proxy, not the published H2O algorithm.

The latest numerical contribution is more careful than “the solver is exact.” In the sequential synthetic audit the original procedure completed 66/80 trajectories; a checking-and-rescue wrapper completed all 5,120 scheduled operations with eight initialization rescues and 84 update rescues. The largest reported readout discrepancy was 0.5273% of retained-value range. This supports a finite evaluated numerical policy, not unrestricted robustness on long model-key streams. [Current SV Attention paper](https://arxiv.org/html/2607.12204v4).

## GemmaSV: a useful integration reference, not a compression result

GemmaSV replaces the global attention operator in frozen Gemma 3. The canonical evaluated setting uses `nu=0.7`, 128-token boundaries, deterministic bandwidth sampling, and prefix-mass preservation. At 4B, all five global layers are grafted; local sliding-window layers remain native. The gate changes attention weights; selection followed by ordinary softmax is therefore a substantive method change.

The scale check records paired PPL changes of +3.704%, +1.851%, and +11.699% at 1B, 4B, and 12B. Strict admission goes from 7/8 to 3/8, 6/8 to 6/8, and 6/8 to 5/8 respectively. Thus 4B is the strongest tested transfer point, not evidence that the same gate works across scales. At 4B the historical local certificate's maximum probe KL is 6.48e-11, but 66.4% of affected decrements need refit fallback. These local refits preserve already-contextualized surviving rows; they are distinct from rebuilding the conversation without the deleted record. [GemmaSV v4](https://arxiv.org/html/2607.27539v4).

The later matched-cost study finds no speed advantage for the current FP32 proxy: its paired update ratio is 1.012 versus graft rebuild and 35.79 versus base rebuild. This is a workload involving copied caches, masks, gates, and teacher-forced probes, not a compressed cache decoder. In particular, logical masks still retain the K/V arrays. Logical masking therefore does not establish physical cache-memory savings. See the public repository's [study description](https://github.com/VyLabs-AI/gemmasv).

## The exactness theorem does not transfer to ordinary softmax

Evicting an SVDD zero coefficient is exactly output-neutral for the earlier **coefficient-weighted** readout: its numerator and denominator both multiply each key's contribution by its coefficient. The result does not hold for ordinary softmax over retained keys, where every finite, unmasked score has strictly positive weight.

A direct counterexample uses scalar keys `(-1, +1, 0)` and values `(0, 0, 1)`. Under a sufficiently broad RBF SVDD fit, the middle key is reserve and the endpoint coefficients are `(1/2, 1/2)`. For a dot-product query `q=0`, ordinary softmax gives all three keys weight `1/3`: the output is `1/3`. Dropping the reserve key changes it to `0`. Query independence of the selector does not change this result.

For ordinary softmax, the useful exact identity instead is:

```
o_full = (1 - delta) * o_kept + delta * o_dropped
o_full - o_kept = delta * (o_dropped - o_kept)
```

Here `delta` is the discarded **softmax attention mass for the particular query**, and the two subset outputs are separately normalized. If all values have norm at most `Vmax`, the error is at most `2 * Vmax * delta`. This is an elementary identity and bound derived here, not a new SVDD theorem. A small SVDD coefficient alone does not bound `delta`.

The pilot evaluates **query-independent geometric cache selection through measured quality and cost**. The earlier reactivation construction identifies a limitation of streaming support sets, but supplies no guarantee for an ordinary-softmax cache.

## Scaling and budget corrections

- `nu` is not the retained fraction. Since coefficients sum to one and each is at most `1/(nu*n)`, the support count is at least `ceil(nu*n)` in exact arithmetic, and may be much larger. Decreasing `nu` permits more sparsity; it does not guarantee a 10% cache. Realized support fractions and exact-budget rules are separate quantities.
- Keeping the largest coefficients when support exceeds a budget is a further heuristic. Capped coefficients often tie, making tie handling and ordering relevant. Top-k truncation is not exact support-set pruning.
- A full 32,768-token Gram matrix has 1,073,741,824 entries: **4 GiB in fp32 per head**, or 8 GiB in fp64, before solver temporaries. Total overhead multiplies across concurrently solved heads and layers.
- A 128-token *fit interval* is not a 128-token *problem*. GemmaSV solves the whole prefix at each boundary. Its batched causal code can allocate `Kpad` with shape `(number_of_boundaries * grouped_heads, max_prefix, max_prefix)`. Reusing that code unchanged does not fix the long-context scaling problem.
- A blockwise approximation, coreset, or feature approximation changes the selection problem; approximation error is distinct from numerical solver error. Local block boundaries can bias selection; equal per-block quotas prevent adaptive allocation across blocks.
- Compacting already-RoPE-transformed keys requires preserving absolute positions, native softmax scaling, query/key normalization, GQA structure, and mask semantics. Context-token budget percentages are distinct from model weights, protected recent tokens, decode growth, and temporary solver memory.

## Implementation distinctions

The earlier implementations have different computational and numerical scopes:

- Full QP solvers and incremental bordered inverses support small-kernel numerical audits. The v4 SV Attention paper documents failure modes and checking-and-rescue policies; these are not long-context serving algorithms.
- Batched FISTA and capped-simplex projection produce approximate fp32 coefficients and still require dense Gram matrices. They do not inherit an exact-arithmetic deletion certificate.
- The earlier TinyGPT selection prototype uses dense masks and retains full cache storage. Its similarity-mass proxy also accesses sequence queries beyond a causal selection boundary. It is not a production KV baseline.
- The synthetic causal-selection experiments use an RBF toy readout and a warm-query proxy, rather than an LLM KV benchmark.
- GemmaSV provides integration evidence for RoPE, grouped-query attention, and cached sessions, while changing the attention operator and retaining prefix-fit costs.
- A support-set budget can exceed its nominal size when too many active points remain. The pilot instead imposes an exact retained-position count.

These distinctions motivate separate measurements of quality, physical cache size, and selector cost. The prior results alone establish neither a compression-performance claim nor certified softmax eviction. The pilot's measurements and limitations are reported in the [results overview](results.md).
