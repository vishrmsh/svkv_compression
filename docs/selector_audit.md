# Selector implementation audit

This audit checks whether poor SVDD results could follow from a
solver or cache-selection implementation mistake. It does not tune the pilot
or establish that other SVDD designs would behave similarly. The deployed
scorer and inference runtime were unchanged during this audit.

## Independent constrained optimization check

The new test `test_svdd_matches_independent_constrained_qp_and_kkt` compares
normalized coefficients from the actual scorer with an independent SciPy
SLSQP solution of

`minimize alpha.T @ K @ alpha`, subject to `sum(alpha)=1` and
`0 <= alpha_i <= 1/(nu*N)`.

For the RBF kernel, `K_ii=1`, so dropping the constant diagonal term makes
this the SVDD dual. The test uses 40 nonidentical, anisotropically distributed
five-dimensional keys and the deployed normalization/bandwidth procedure.

| Check | Deployed `nu=0.05` | Test-only `nu=0.4` |
| --- | ---: | ---: |
| Absolute objective difference from constrained reference | `1.17e-10` | `5.29e-14` |
| Maximum KKT gradient residual | `3.67e-6` | `5.26e-7` |
| Active upper-bound constraints | 0 | 14 |

The second setting is a unit-test stress case, not a pilot hyperparameter
change. It tests the dual box bound as well as the simplex normalization.
Both cases pass feasibility, objective, interior-stationarity, and bound-sign
checks. All 59 selector tests pass after this addition. These checks support
the OneClassSVM-to-SVDD conversion used in each block; they do not turn the
blockwise ranker into a global solution.

## Stored-diagnostics checks

The **completed corrected pilot** in `results/pilot` was audited: all 54
contexts and 1,350 prediction rows, with exactly 25 variants per context.
The recorded selector source hash matches the audited implementation. The
numbers below supersede the earlier partial audit of the archived
`results/pre_separator_check` run. For each of the post-RoPE and pre-RoPE
SVDD variants:

- All 399,840 recorded block solves converged, or 799,680 across both variants.
- The largest kernel block contained 65,536 entries (`256 × 256`).
- Raw dual sums matched `nu * block_length` to at most `5.2e-14` absolute error.
- Head-level raw support fractions equaled summed solver support counts
  divided by summed block lengths.
- Selection support fractions matched the corresponding fitted layer.
- Retained plus discarded supports equaled the raw support count; retained
  supports plus retained zero-score positions equaled the exact budget.
- Stored certificate flags correctly disclaimed a global solve and exact
  ordinary-softmax eviction.

| Diagnostic in the completed pilot | Post-RoPE SVDD | Pre-RoPE SVDD |
| --- | ---: | ---: |
| Raw support fraction, mean over fitted layers | 16.03% | 14.87% |
| Raw supports discarded at 10% cache budget | 37.86% | 33.44% |
| Raw supports discarded at 20% cache budget | 4.16% | 1.23% |
| Raw supports discarded at 50% cache budget | 0% | 0% |
| Zero-score share of retained positions at 10% budget | 1.87% | 1.66% |
| Zero-score share of retained positions at 20% budget | 24.34% | 27.05% |
| Zero-score share of retained positions at 50% budget | 68.43% | 70.46% |
| Minimum chronological fill share of retained positions at 50% budget | 68.00% | 70.03% |

Zero-score retained positions include both common sink/recent reservations
and the chronological budget-filling rule. Support-retention percentages are
pooled over positions, whereas the first row averages fitted layer fractions.
The last row subtracts every sink/recent reserved slot from zero-score
retentions, so it is a conservative lower bound on unreserved chronological
fill. Thus the large zero-score share is not explained by shared reservations.

Raw head-level support fractions ranged from 7.41% to 36.05% for post-RoPE
and 9.45% to 35.62% for pre-RoPE. Among full 256-token blocks, the 5th/50th/95th
percentiles of support counts were 26/38/64 and 27/36/52, respectively. These
are materially different support densities under the same `nu=0.05`.

## Algorithmic and experimental boundaries

No asymmetric cache handling or question visibility was found in the runner.
All methods receive the same uncompressed context prefix, compress before
the question, reserve the same four sinks and 32 recent positions, and gather
original key/value rows per KV head. Streaming intentionally spends its full
remaining budget on recency. Original absolute RoPE offsets are retained for
all methods; the pre-RoPE variants change only their scoring representation.
Cached scores are reused across budgets, and decode order is randomized.

The following are genuine design limitations, rather than numerical defects:

- Each SVDD block has its own optimization and fixed total ranking score
  equal to its block length. Ranking those local coefficients globally is a
  heuristic. Blocks with different geometry need not have comparable notions
  of global token usefulness, even after correcting final-block length bias.
  In particular, a block's mean positive score is `block_length / supports`;
  a sparser local support set receives larger positive scores on average.
  The implementation therefore does not select a calibrated global notion of
  rarity or attention relevance.
- At 10% budget the method discards many actual supports. At 50% it keeps all
  fitted supports and fills many remaining slots with zero-score positions.
  Those ties prefer earlier positions, introducing a prefix preference.
  At the 10% budget, however, fewer than 2% of selected positions have zero
  scores. Chronological fill cannot by itself explain the tight-budget
  outcome. That outcome tests truncation by local dual magnitude, while the
  50% result largely tests a mixture of all local supports and prefix filling.
- Unit-normalized SVDD examines angular geometry. Leverage also uses magnitude
  and covariance structure; their difference is intentional, not an identical
  geometric objective solved by different software.
- The post-RoPE SVDD geometry can reflect token positions. The pre-RoPE
  ablation addresses this particular issue without altering serving keys.
- All methods use one-shot compression after full prefill. This gives a fair
  common cache for this comparison, but does not test a bounded-memory stream
  or reproduce every original baseline's execution schedule.
- SnapKV observes context-tail queries, not the evaluation question. Results
  apply to this reusable-cache setting; they do not measure question-aware
  SnapKV's strongest configuration.

These checks rule out the tested solver-normalization, support-accounting,
and shared-cache confounds. They cannot rule out every bug or establish the
best possible SVDD compression design. Poor results should be attributed to
the specified pilot ranker, rather than to SVDD in general. Changing the
cross-block ranking or zero-score tie rule would define a new variant that
needs fresh held-out evaluation; it is not a correction to a failed solve.
