# What the pilot's selectors do

The pilot compares selection of original key/value positions. Scoring uses
post-RoPE keys by default, but leverage uses pre-RoPE keys and the inference
harness may also run an explicitly labeled pre-RoPE SVDD ablation. The
selector never changes the retained cache vectors or their original positions.
It does not multiply attention weights by support-vector coefficients. After
selection, inference uses ordinary softmax attention over the retained KV
entries. Each KV head receives exactly the same number of retained positions,
although its chosen positions can differ from other heads.

## Blockwise SVDD ranker

`svdd` is a practical approximation to test the geometric-selection hypothesis.
It is **not a global SVDD solve**, an optimized streaming implementation, or a
certified softmax eviction policy.

1. L2-normalize each key. A zero key remains zero. Normalization is used only
   for scoring; the model retains its original key and value vectors.
2. Independently for each KV head, sample at most 256 keys using a fixed seed
   without replacement. Take the median of strictly positive squared pairwise
   distances, and set RBF `gamma = 1 / median`. Use `gamma = 1` if all sampled
   distances are zero. This bandwidth is shared by all blocks within the head.
3. Partition positions into chronological blocks of at most 256 tokens. Fit
   scikit-learn `OneClassSVM(kernel="rbf", nu=0.05, tol=1e-5,
   max_iter=10000)` separately to each block. Record convergence and iteration
   counts. Incomplete solves remain visible in the result diagnostics.
4. Divide each block's raw positive dual coefficients by their sum. For a
   unit-diagonal RBF kernel, these normalized coefficients solve the equivalent
   SVDD simplex problem up to numerical solver tolerance. Assign zero to
   positions outside the solver's support set.
5. Rank using `block_length * normalized_dual`. Multiplication by block length
   removes the automatic `1 / block_length` inflation of a final partial
   block. It does not make local coefficients into a globally optimal solution.
6. Keep the highest-ranked positions after reserving initial/recent positions.
   If there are too many supports, discard lower-scoring supports. If too few,
   fill the remaining budget with zero-score positions, breaking ties by earlier
   original position. Always report the raw support fraction and how many
   positive-coefficient positions were discarded.

`nu=0.05` is fixed across budgets and evaluation examples; it is not tuned using
task labels. It lower-bounds support fraction within each fitted block, up to
numerical tolerance, but does **not** request a 5% retained cache. The exact
cache budget is imposed by step 6. A narrow RBF bandwidth or highly dispersed
keys can make nearly every key a support even at small `nu`.

Only block-sized kernel problems and a 256-key bandwidth sample are required.
The algorithm does not build a context-sized `N × N` Gram matrix. This bounds
the kernel problem size but does not imply low end-to-end latency: every
layer/head/block invokes a CPU solver and the pilot transfers keys to NumPy.
The score arrays and normalized key arrays still scale linearly with context.

## Fixed decision-score follow-up

`svdd_decision` is a separately reported follow-up on the same 36 needle cases.
It uses exactly the original post-RoPE keys, normalization, RBF bandwidth,
256-token fits, `nu=0.05`, solver tolerance, and common cache reserves. The only
algorithmic change is the ranking score. With raw OneClassSVM dual mass `s`,
normalized coefficients `alpha`, and normalized offset `rho`, the score is

```
score(x) = -2 * decision_function(x) / s
         = 2 * rho - 2 * sum_i alpha_i * K(x_i, x)
         = squared_feature_distance_to_center(x) - squared_radius
```

Higher scores retain keys farther outside their fitted block's sphere. Dividing
by the actual dual mass removes LIBSVM's `nu * block_length` scale, including
the final partial block. The raw signed scores are ranked **globally across all
positions in each head**. There are no per-block retention quotas, percentile
transforms, or z-scores. Each block still has its own fitted center and radius;
this does not become a global SVDD solve. The original alpha method also used
global ranking across blocks.

A free support vector lies on the fitted boundary, so decision scoring does
not remove every tie. Solver-scale variation near zero can affect the ordering
of these keys. The frozen follow-up neither snaps scores to zero nor tunes a
tie threshold; it retains the existing stable exact-score tie rule. Diagnostics
record free supports' boundary residuals. Positive decision scores do not mean
positive dual coefficients, so support counts come from the fitted support set
and never from the signs of decision scores.

The setting was fixed before this follow-up's inference and is stored in
[`configs/decision_followup.json`](../configs/decision_followup.json). These
already examined cases can diagnose the original failure; they cannot establish
a held-out improvement. The original pilot and its execution source remain
unchanged under `results/pilot`.

## Baseline scores and shared budget rules

| Method | Larger score means higher retention priority |
| --- | --- |
| `keydiff` | Negative cosine between the raw key and the mean of L2-normalized keys. This matches the scoring formula in NVIDIA kvpress; the pilot is a one-shot selection comparison, not KeyDiff's iterative blockwise schedule. |
| `leverage` | Approximate ridge leverage of centered **pre-RoPE** keys, using a seeded right Gaussian sketch with dimension `min(48, head_dim)` and ridge `0.01`. See details below. |
| `knorm` | Inverse raw key L2 norm, with a numerical floor for zero norm. |
| `random` | Seeded independent uniform random score per head and token. |
| `snapkv` | Model-derived observation-window scores supplied by the inference harness. The scorer here does not approximate attention with geometry. |
| `streaming` | Initial sink positions plus the newest remaining positions that fit the budget. This is a sink/recent retention baseline; a one-shot prefill compression does not reproduce StreamingLLM's full streaming execution schedule. |

All ranked methods reserve the first four positions and the final 32 positions,
included in the stated total budget. Streaming reserves the first four and
spends the remaining budget on the newest positions. If a budget is smaller
than 36, sinks have priority and then the newest positions that fit are kept.
Equal ranking scores prefer earlier positions. Selected indices are returned
chronologically; original RoPE positions must be preserved by the harness.

The APIs are `compute_scores(keys, method, seed=0)` and
`select_indices(keys, budget, method, scores=None, seed=0, recent=32, sinks=4)`.
Keys have shape `[KV_heads, tokens, head_dim]`. Score computation can be reused
across budgets. For SnapKV, supply precomputed `[KV_heads, tokens]` scores to
`select_indices`. When precomputed scores are supplied, preserve the original
score diagnostics alongside selection diagnostics; selection cannot recover
the solver's convergence history from the scores alone.

## Leverage baseline

The `leverage` scorer follows the geometry-based `LeverageScorePress` formula
in NVIDIA kvpress. The caller supplies pre-RoPE keys. For each head, center
the keys across sequence positions, then form `X = centered_keys @ Phi`,
where `Phi` has independent Gaussian entries of variance `1 / k` and
`k = min(48, head_dim)`. With
`L = cholesky(X.T @ X + 0.01 * I)`, the score is the squared column norm of
`solve_triangular(L, X.T)`. This equals the diagonal of the ridge hat matrix
`X @ inverse(X.T @ X + 0.01 * I) @ X.T` without constructing that sequence-sized
matrix or an explicit inverse. Larger scores favor retention.

The pilot uses NumPy's seeded generator and float64 arithmetic. It omits the
reference implementation's final global z-score transform, which does not
change within-head rankings. Rank is capped at the key dimension for small
heads. Diagnostics record the sketch dimension, ridge, and score time. This
is the leverage component alone, not the blended Compactor policy.

Tests compare against a small explicit ridge hat matrix, verify translation
invariance and token-permutation equivariance, and check rare-direction
retention and finite scores for constant keys.

## The certificate does not transfer to ordinary softmax

The earlier support-weighted memory's zero-coefficient deletion result does
not certify eviction from an ordinary softmax KV cache. With logits `q·k_i`,
every finite logit has positive softmax weight, including keys with zero SVDD
coefficient. Deleting such a key generally changes both the numerator and
denominator of attention immediately.

A minimal counterexample is ten identical keys, query zero, one value equal
to one, and nine values zero. Full attention returns `0.1`. An SVDD optimum
can give the value-one key coefficient zero because its key is redundant in
the geometric objective. Removing that key changes attention to zero. The
executable test `test_zero_dual_does_not_certify_ordinary_softmax_eviction`
checks this construction against the actual solver. Retaining every fitted
support is therefore insufficient to claim exact ordinary-softmax preservation,
even at the instant of compression. A blockwise solution or budget truncation
adds further approximation; neither repairs this issue.

The test `test_distinct_key_softmax_counterexample` also checks a nonduplicate
example: keys `(-1, 0, 1)`, RBF `gamma=1/4`, and `nu=1/2` give normalized
coefficients `(1/2, 0, 1/2)`. Query zero and values `(0, 1, 0)` yield full
softmax output `1/3`, versus zero after support-only selection. This separate
theory test does not normalize the scalar keys and is not a pilot setting.

## Source references

- [NVIDIA kvpress KeyDiff score implementation](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/keydiff_press.py)
- [NVIDIA kvpress leverage score implementation](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/leverage_press.py)
- [scikit-learn OneClassSVM API and solver attributes](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)

The baseline implementation above was independently written from the stated
formulas. No code or files in `plain_jane` are modified.
