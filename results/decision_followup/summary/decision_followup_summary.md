# Fixed SVDD decision-score follow-up

**36 reused needle cases; 360 new answers.** The original alpha results are a separate historical comparator. No rows are pooled across runs.

This changes ranking from normalized alpha coefficients to signed radius excess while preserving the original block fits. Both rankers select globally across blocks within each head; neither assigns equal per-block retention quotas. A blockwise score is not a global SVDD solve. The selector remains query-independent and uses ordinary softmax after physical compaction.

These same 36 custom needle cases were inspected in the original pilot. This is a fixed diagnostic follow-up, **not held-out evidence** and not official RULER. Matching a cheaper baseline cannot establish a reason to reopen the paper.

## Mean numeric recall

| Method | 10% KV | 20% KV | 50% KV | Cases per budget |
|---|---:|---:|---:|---:|
| Original SVDD alpha | 0.0694 | 0.3194 | 0.5833 | 36 |
| SVDD decision score | 0.0556 | 0.4583 | 1.0000 | 36 |
| KeyDiff | 0.9861 | 1.0000 | 1.0000 | 36 |
| Leverage | 0.9722 | 1.0000 | 1.0000 | 36 |

Contemporaneous full-cache recall: **1.0000** on 36 cases. Each multi-needle case receives mean numeric recall across its two gold numbers; extra guesses are not penalized.

### By task

| Task | Method | 10% KV | 20% KV | 50% KV | Cases per budget |
|---|---|---:|---:|---:|---:|
| niah_multi | Original SVDD alpha | 0.1389 | 0.4167 | 0.6667 | 18 |
| niah_multi | SVDD decision score | 0.0000 | 0.5833 | 1.0000 | 18 |
| niah_multi | KeyDiff | 0.9722 | 1.0000 | 1.0000 | 18 |
| niah_multi | Leverage | 0.9444 | 1.0000 | 1.0000 | 18 |
| niah_single | Original SVDD alpha | 0.0000 | 0.2222 | 0.5000 | 18 |
| niah_single | SVDD decision score | 0.1111 | 0.3333 | 1.0000 | 18 |
| niah_single | KeyDiff | 1.0000 | 1.0000 | 1.0000 | 18 |
| niah_single | Leverage | 1.0000 | 1.0000 | 1.0000 | 18 |

### By nominal context tokens

| Nominal context tokens | Method | 10% KV | 20% KV | 50% KV | Cases per budget |
|---|---|---:|---:|---:|---:|
| 8192 | Original SVDD alpha | 0.0417 | 0.2917 | 0.4583 | 12 |
| 8192 | SVDD decision score | 0.0000 | 0.4583 | 1.0000 | 12 |
| 8192 | KeyDiff | 1.0000 | 1.0000 | 1.0000 | 12 |
| 8192 | Leverage | 0.9167 | 1.0000 | 1.0000 | 12 |
| 16384 | Original SVDD alpha | 0.0833 | 0.2500 | 0.5833 | 12 |
| 16384 | SVDD decision score | 0.0000 | 0.3333 | 1.0000 | 12 |
| 16384 | KeyDiff | 1.0000 | 1.0000 | 1.0000 | 12 |
| 16384 | Leverage | 1.0000 | 1.0000 | 1.0000 | 12 |
| 32768 | Original SVDD alpha | 0.0833 | 0.4167 | 0.7083 | 12 |
| 32768 | SVDD decision score | 0.1667 | 0.5833 | 1.0000 | 12 |
| 32768 | KeyDiff | 0.9583 | 1.0000 | 1.0000 | 12 |
| 32768 | Leverage | 1.0000 | 1.0000 | 1.0000 | 12 |

## Paired differences

Decision-score recall minus the comparator, with 2,000 paired case-bootstrap replicates (seed `20260927`). Intervals are exploratory, unadjusted for multiple comparisons, and do not account for dependence among cases sharing filler or seeds.

| Comparator | Retention | Mean difference | Bootstrap 95% interval | Wins / ties / losses |
|---|---:|---:|---|---|
| Full cache | 10% | -0.9444 | [-1.0000, -0.8611] | 0 / 2 / 34 |
| KeyDiff | 10% | -0.9306 | [-1.0000, -0.8472] | 0 / 2 / 34 |
| Leverage | 10% | -0.9167 | [-0.9861, -0.8194] | 0 / 2 / 34 |
| Original SVDD alpha | 10% | -0.0139 | [-0.1111, +0.0972] | 2 / 29 / 5 |
| Full cache | 20% | -0.5417 | [-0.6667, -0.4167] | 0 / 10 / 26 |
| KeyDiff | 20% | -0.5417 | [-0.6806, -0.4167] | 0 / 10 / 26 |
| Leverage | 20% | -0.5417 | [-0.6806, -0.4028] | 0 / 10 / 26 |
| Original SVDD alpha | 20% | +0.1389 | [+0.0417, +0.2500] | 9 / 26 / 1 |
| Full cache | 50% | +0.0000 | [+0.0000, +0.0000] | 0 / 36 / 0 |
| KeyDiff | 50% | +0.0000 | [+0.0000, +0.0000] | 0 / 36 / 0 |
| Leverage | 50% | +0.0000 | [+0.0000, +0.0000] | 0 / 36 / 0 |
| Original SVDD alpha | 50% | +0.4167 | [+0.2917, +0.5417] | 21 / 15 / 0 |

## Shared-harness scoring-stage wall time

| Nominal context tokens | SVDD decision score (s) | KeyDiff (s) | Leverage (s) | Cases |
|---:|---:|---:|---:|---:|
| 8,192 | 5.117 | 0.227 | 0.345 | 12 |
| 16,384 | 10.199 | 0.480 | 0.720 | 12 |
| 32,768 | 19.742 | 0.954 | 1.447 | 12 |

The scoring stage runs once per case and is reused across budgets; these are per-case mean stage times. They include the layer loop and SVDD diagnostic JSON serialization/output, and leverage includes inverse RoPE. Key transfer and physical compaction are outside this timer. These are measured harness costs, not isolated deployment timings or pure arithmetic benchmarks.

## Independent validation

All expected arms are present exactly once; hashes, gold answers, rescored answers, and exact KV byte accounting passed. 297,024 block fits were recorded; 0 did not converge.

Against the original post-RoPE alpha run, 1,813,392 recorded fit field values were compared exactly across 297,024 blocks: **0 differences**. This includes bandwidths, support counts, iteration counts, and raw dual sums. These are fit-metadata checks, not a comparison of archived full coefficient arrays.

Repeated full/KeyDiff/leverage arms agree with the original scores in **252/252** comparisons and with exact output text in **252/252**. Every comparison is retained in `baseline_agreement.csv`.

Raw solver supports describe the fitted dual only. Positive signed decision scores describe points outside a fitted boundary; they are not support coefficients and are never reported as supports retained.

The fits contain 11,354,269 free support coordinates (95.52% of all positive-dual support coordinates). Of those, 11,354,269 lie within the recorded block boundary tolerance; the largest absolute free-support decision score is 1.21e-06. Free supports can therefore remain tied at the fitted boundary within solver precision. This diagnostic does not establish that boundary ties caused a particular retrieval failure.

`aggregate_recall.csv` and `paired_deltas.csv` include task, context-length, and combined task-by-length breakdowns. `validation.json` and `summary_manifest.json` record checks and provenance. The figure is available as PNG and PDF.

Selection timings in these answer rows come from the shared quality harness; they are not an isolated deployment latency benchmark. The original isolated systems measurements do not measure this new decision-score implementation.
