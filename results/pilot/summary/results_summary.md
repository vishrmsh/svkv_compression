# Exploratory KV-compression pilot results

Input: `results/pilot/predictions.jsonl`. **54 cases, 1350 unique prediction rows**, 5 tasks. Methods present: full, keydiff, knorm, leverage, random, snapkv, streaming, svdd, svdd_prerope.

Coverage: **0 expected rows missing among observed cases**. This count cannot detect cases absent from the prediction file altogether; compare with the case manifest. Identical duplicate lines ignored: 0. See `missing_predictions.csv`.

These are small, exploratory results on a local runtime. Custom NIAH tasks are **not official RULER**. LongBench rows are a selected small subset. Different task scores are reported separately; a mixed-task average is not a benchmark score.

Compression occurs after full-context prefill and **before the downstream question**. SnapKV observes the context tail with the question hidden. This measures context-only selection and is not a reproduction of question-visible SnapKV or an online streaming policy.

## Task scores

| Task | Method | Nominal retention | Cases | Mean score | Prefix KV MiB | Compression overhead (s) | Estimated total (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| hotpotqa | Full cache | 100% | 6 | 0.313 | 855.4 | 0.000 | 12.513 |
| hotpotqa | KeyDiff | 10% | 6 | 0.194 | 85.5 | 0.613 | 12.680 |
| hotpotqa | KeyDiff | 20% | 6 | 0.211 | 171.1 | 0.617 | 12.850 |
| hotpotqa | KeyDiff | 50% | 6 | 0.251 | 427.7 | 0.628 | 13.168 |
| hotpotqa | K-norm | 10% | 6 | 0.055 | 85.5 | 0.272 | 12.489 |
| hotpotqa | K-norm | 20% | 6 | 0.019 | 171.1 | 0.276 | 12.512 |
| hotpotqa | K-norm | 50% | 6 | 0.070 | 427.7 | 0.289 | 12.580 |
| hotpotqa | Leverage only | 10% | 6 | 0.249 | 85.5 | 0.842 | 12.919 |
| hotpotqa | Leverage only | 20% | 6 | 0.194 | 171.1 | 0.845 | 13.064 |
| hotpotqa | Leverage only | 50% | 6 | 0.297 | 427.7 | 0.857 | 13.310 |
| hotpotqa | Random | 10% | 6 | 0.059 | 85.5 | 0.180 | 12.656 |
| hotpotqa | Random | 20% | 6 | 0.071 | 171.1 | 0.188 | 12.732 |
| hotpotqa | Random | 50% | 6 | 0.140 | 427.7 | 0.196 | 12.589 |
| hotpotqa | SnapKV (question hidden) | 10% | 6 | 0.259 | 85.5 | 0.176 | 12.197 |
| hotpotqa | SnapKV (question hidden) | 20% | 6 | 0.206 | 171.1 | 0.182 | 12.531 |
| hotpotqa | SnapKV (question hidden) | 50% | 6 | 0.315 | 427.7 | 0.196 | 12.409 |
| hotpotqa | Initial + recent | 10% | 6 | 0.057 | 85.5 | 0.020 | 12.253 |
| hotpotqa | Initial + recent | 20% | 6 | 0.121 | 171.1 | 0.023 | 12.341 |
| hotpotqa | Initial + recent | 50% | 6 | 0.179 | 427.7 | 0.022 | 12.421 |
| hotpotqa | SVDD post-RoPE | 10% | 6 | 0.066 | 85.5 | 4.515 | 16.974 |
| hotpotqa | SVDD post-RoPE | 20% | 6 | 0.238 | 171.1 | 4.519 | 16.612 |
| hotpotqa | SVDD post-RoPE | 50% | 6 | 0.300 | 427.7 | 4.533 | 16.681 |
| hotpotqa | SVDD pre-RoPE | 10% | 6 | 0.025 | 85.5 | 4.725 | 17.304 |
| hotpotqa | SVDD pre-RoPE | 20% | 6 | 0.255 | 171.1 | 4.728 | 17.081 |
| hotpotqa | SVDD pre-RoPE | 50% | 6 | 0.259 | 427.7 | 4.743 | 16.955 |
| niah_multi | Full cache | 100% | 18 | 1.000 | 1031.3 | 0.000 | 16.035 |
| niah_multi | KeyDiff | 10% | 18 | 0.972 | 103.1 | 0.726 | 16.565 |
| niah_multi | KeyDiff | 20% | 18 | 1.000 | 206.2 | 0.732 | 16.593 |
| niah_multi | KeyDiff | 50% | 18 | 1.000 | 515.7 | 0.744 | 16.661 |
| niah_multi | K-norm | 10% | 18 | 0.000 | 103.1 | 0.316 | 16.108 |
| niah_multi | K-norm | 20% | 18 | 0.000 | 206.2 | 0.321 | 16.171 |
| niah_multi | K-norm | 50% | 18 | 0.056 | 515.7 | 0.336 | 16.214 |
| niah_multi | Leverage only | 10% | 18 | 0.944 | 103.1 | 1.013 | 16.853 |
| niah_multi | Leverage only | 20% | 18 | 1.000 | 206.2 | 1.015 | 16.874 |
| niah_multi | Leverage only | 50% | 18 | 1.000 | 515.7 | 1.028 | 16.945 |
| niah_multi | Random | 10% | 18 | 0.000 | 103.1 | 0.211 | 16.014 |
| niah_multi | Random | 20% | 18 | 0.000 | 206.2 | 0.215 | 16.051 |
| niah_multi | Random | 50% | 18 | 0.167 | 515.7 | 0.230 | 16.143 |
| niah_multi | SnapKV (question hidden) | 10% | 18 | 0.000 | 103.1 | 0.199 | 15.980 |
| niah_multi | SnapKV (question hidden) | 20% | 18 | 0.111 | 206.2 | 0.204 | 16.027 |
| niah_multi | SnapKV (question hidden) | 50% | 18 | 0.278 | 515.7 | 0.217 | 16.114 |
| niah_multi | Initial + recent | 10% | 18 | 0.000 | 103.1 | 0.021 | 15.797 |
| niah_multi | Initial + recent | 20% | 18 | 0.333 | 206.2 | 0.022 | 15.852 |
| niah_multi | Initial + recent | 50% | 18 | 0.500 | 515.7 | 0.025 | 15.942 |
| niah_multi | SVDD post-RoPE | 10% | 18 | 0.139 | 103.1 | 5.351 | 21.180 |
| niah_multi | SVDD post-RoPE | 20% | 18 | 0.417 | 206.2 | 5.356 | 21.215 |
| niah_multi | SVDD post-RoPE | 50% | 18 | 0.667 | 515.7 | 5.373 | 21.288 |
| niah_multi | SVDD pre-RoPE | 10% | 18 | 0.139 | 103.1 | 5.649 | 21.486 |
| niah_multi | SVDD pre-RoPE | 20% | 18 | 0.361 | 206.2 | 5.655 | 21.514 |
| niah_multi | SVDD pre-RoPE | 50% | 18 | 0.611 | 515.7 | 5.670 | 21.587 |
| niah_single | Full cache | 100% | 18 | 1.000 | 1031.3 | 0.000 | 15.881 |
| niah_single | KeyDiff | 10% | 18 | 1.000 | 103.1 | 0.734 | 16.488 |
| niah_single | KeyDiff | 20% | 18 | 1.000 | 206.2 | 0.739 | 16.503 |
| niah_single | KeyDiff | 50% | 18 | 1.000 | 515.7 | 0.755 | 16.558 |
| niah_single | K-norm | 10% | 18 | 0.000 | 103.1 | 0.326 | 16.174 |
| niah_single | K-norm | 20% | 18 | 0.000 | 206.2 | 0.329 | 16.233 |
| niah_single | K-norm | 50% | 18 | 0.056 | 515.7 | 0.343 | 16.300 |
| niah_single | Leverage only | 10% | 18 | 1.000 | 103.1 | 1.021 | 16.772 |
| niah_single | Leverage only | 20% | 18 | 1.000 | 206.2 | 1.025 | 16.790 |
| niah_single | Leverage only | 50% | 18 | 1.000 | 515.7 | 1.039 | 16.841 |
| niah_single | Random | 10% | 18 | 0.000 | 103.1 | 0.218 | 15.988 |
| niah_single | Random | 20% | 18 | 0.000 | 206.2 | 0.221 | 16.009 |
| niah_single | Random | 50% | 18 | 0.167 | 515.7 | 0.235 | 16.144 |
| niah_single | SnapKV (question hidden) | 10% | 18 | 0.000 | 103.1 | 0.198 | 16.003 |
| niah_single | SnapKV (question hidden) | 20% | 18 | 0.056 | 206.2 | 0.203 | 16.014 |
| niah_single | SnapKV (question hidden) | 50% | 18 | 0.444 | 515.7 | 0.216 | 16.106 |
| niah_single | Initial + recent | 10% | 18 | 0.000 | 103.1 | 0.022 | 15.794 |
| niah_single | Initial + recent | 20% | 18 | 0.333 | 206.2 | 0.023 | 15.830 |
| niah_single | Initial + recent | 50% | 18 | 0.667 | 515.7 | 0.025 | 15.914 |
| niah_single | SVDD post-RoPE | 10% | 18 | 0.000 | 103.1 | 5.393 | 21.189 |
| niah_single | SVDD post-RoPE | 20% | 18 | 0.222 | 206.2 | 5.402 | 21.205 |
| niah_single | SVDD post-RoPE | 50% | 18 | 0.500 | 515.7 | 5.416 | 21.243 |
| niah_single | SVDD pre-RoPE | 10% | 18 | 0.056 | 103.1 | 5.655 | 21.416 |
| niah_single | SVDD pre-RoPE | 20% | 18 | 0.222 | 206.2 | 5.659 | 21.463 |
| niah_single | SVDD pre-RoPE | 50% | 18 | 0.444 | 515.7 | 5.677 | 21.539 |
| passage_retrieval_en | Full cache | 100% | 6 | 1.000 | 694.3 | 0.000 | 9.271 |
| passage_retrieval_en | KeyDiff | 10% | 6 | 0.167 | 69.4 | 0.487 | 9.683 |
| passage_retrieval_en | KeyDiff | 20% | 6 | 0.833 | 138.8 | 0.490 | 9.689 |
| passage_retrieval_en | KeyDiff | 50% | 6 | 1.000 | 347.1 | 0.501 | 9.729 |
| passage_retrieval_en | K-norm | 10% | 6 | 0.167 | 69.4 | 0.211 | 9.513 |
| passage_retrieval_en | K-norm | 20% | 6 | 0.000 | 138.8 | 0.215 | 9.473 |
| passage_retrieval_en | K-norm | 50% | 6 | 0.500 | 347.1 | 0.222 | 9.446 |
| passage_retrieval_en | Leverage only | 10% | 6 | 0.667 | 69.4 | 0.678 | 9.869 |
| passage_retrieval_en | Leverage only | 20% | 6 | 0.833 | 138.8 | 0.680 | 9.881 |
| passage_retrieval_en | Leverage only | 50% | 6 | 1.000 | 347.1 | 0.691 | 9.917 |
| passage_retrieval_en | Random | 10% | 6 | 0.167 | 69.4 | 0.143 | 9.446 |
| passage_retrieval_en | Random | 20% | 6 | 0.167 | 138.8 | 0.141 | 9.429 |
| passage_retrieval_en | Random | 50% | 6 | 0.833 | 347.1 | 0.151 | 9.382 |
| passage_retrieval_en | SnapKV (question hidden) | 10% | 6 | 0.500 | 69.4 | 0.137 | 9.352 |
| passage_retrieval_en | SnapKV (question hidden) | 20% | 6 | 0.833 | 138.8 | 0.140 | 9.362 |
| passage_retrieval_en | SnapKV (question hidden) | 50% | 6 | 1.000 | 347.1 | 0.150 | 9.376 |
| passage_retrieval_en | Initial + recent | 10% | 6 | 0.000 | 69.4 | 0.016 | 9.321 |
| passage_retrieval_en | Initial + recent | 20% | 6 | 0.000 | 138.8 | 0.016 | 9.334 |
| passage_retrieval_en | Initial + recent | 50% | 6 | 0.000 | 347.1 | 0.016 | 9.378 |
| passage_retrieval_en | SVDD post-RoPE | 10% | 6 | 0.167 | 69.4 | 3.696 | 12.942 |
| passage_retrieval_en | SVDD post-RoPE | 20% | 6 | 0.667 | 138.8 | 3.699 | 12.898 |
| passage_retrieval_en | SVDD post-RoPE | 50% | 6 | 1.000 | 347.1 | 3.709 | 12.934 |
| passage_retrieval_en | SVDD pre-RoPE | 10% | 6 | 0.500 | 69.4 | 3.865 | 13.088 |
| passage_retrieval_en | SVDD pre-RoPE | 20% | 6 | 0.833 | 138.8 | 3.869 | 13.068 |
| passage_retrieval_en | SVDD pre-RoPE | 50% | 6 | 1.000 | 347.1 | 3.879 | 13.104 |
| qasper | Full cache | 100% | 6 | 0.190 | 575.7 | 0.000 | 8.370 |
| qasper | KeyDiff | 10% | 6 | 0.054 | 57.5 | 0.412 | 8.032 |
| qasper | KeyDiff | 20% | 6 | 0.118 | 115.1 | 0.416 | 8.258 |
| qasper | KeyDiff | 50% | 6 | 0.200 | 287.8 | 0.423 | 8.716 |
| qasper | K-norm | 10% | 6 | 0.053 | 57.5 | 0.183 | 7.914 |
| qasper | K-norm | 20% | 6 | 0.049 | 115.1 | 0.187 | 7.974 |
| qasper | K-norm | 50% | 6 | 0.139 | 287.8 | 0.193 | 8.171 |
| qasper | Leverage only | 10% | 6 | 0.067 | 57.5 | 0.570 | 8.444 |
| qasper | Leverage only | 20% | 6 | 0.151 | 115.1 | 0.572 | 8.609 |
| qasper | Leverage only | 50% | 6 | 0.190 | 287.8 | 0.582 | 8.756 |
| qasper | Random | 10% | 6 | 0.086 | 57.5 | 0.125 | 8.378 |
| qasper | Random | 20% | 6 | 0.143 | 115.1 | 0.127 | 8.462 |
| qasper | Random | 50% | 6 | 0.206 | 287.8 | 0.133 | 8.545 |
| qasper | SnapKV (question hidden) | 10% | 6 | 0.047 | 57.5 | 0.120 | 7.835 |
| qasper | SnapKV (question hidden) | 20% | 6 | 0.178 | 115.1 | 0.122 | 8.175 |
| qasper | SnapKV (question hidden) | 50% | 6 | 0.186 | 287.8 | 0.130 | 8.513 |
| qasper | Initial + recent | 10% | 6 | 0.103 | 57.5 | 0.015 | 8.253 |
| qasper | Initial + recent | 20% | 6 | 0.119 | 115.1 | 0.014 | 8.390 |
| qasper | Initial + recent | 50% | 6 | 0.205 | 287.8 | 0.017 | 8.614 |
| qasper | SVDD post-RoPE | 10% | 6 | 0.085 | 57.5 | 3.113 | 11.191 |
| qasper | SVDD post-RoPE | 20% | 6 | 0.224 | 115.1 | 3.113 | 10.919 |
| qasper | SVDD post-RoPE | 50% | 6 | 0.164 | 287.8 | 3.125 | 11.022 |
| qasper | SVDD pre-RoPE | 10% | 6 | 0.092 | 57.5 | 3.199 | 11.261 |
| qasper | SVDD pre-RoPE | 20% | 6 | 0.140 | 115.1 | 3.201 | 11.228 |
| qasper | SVDD pre-RoPE | 50% | 6 | 0.161 | 287.8 | 3.211 | 11.168 |

## Paired SV comparisons

Intervals below use **2,000 paired case-bootstrap replicates**, seed `20260927`, percentile 95% intervals. The paired sample is the intersection of available case IDs at the same retention budget; full cache is compared at 100%. Positive deltas favor the SV method. These intervals are exploratory, unadjusted for multiple comparisons, and can be degenerate with tiny samples. Cases sharing filler, seeds, or source documents are not necessarily independent; case-level resampling does not remove that dependence.

| Task | SV method | Baseline | Retention | Pairs | Score delta | Bootstrap 95% interval | Wins / ties / losses |
|---|---|---|---:|---:|---:|---|---|
| hotpotqa | SVDD post-RoPE | Full cache | 10% | 6 | -0.247 | [-0.514, -0.032] | 2 / 0 / 4 |
| hotpotqa | SVDD post-RoPE | KeyDiff | 10% | 6 | -0.128 | [-0.404, 0.040] | 2 / 0 / 4 |
| hotpotqa | SVDD post-RoPE | K-norm | 10% | 6 | 0.011 | [-0.017, 0.045] | 2 / 1 / 3 |
| hotpotqa | SVDD post-RoPE | Leverage only | 10% | 6 | -0.183 | [-0.446, 0.008] | 2 / 1 / 3 |
| hotpotqa | SVDD post-RoPE | Random | 10% | 6 | 0.007 | [-0.003, 0.019] | 3 / 1 / 2 |
| hotpotqa | SVDD post-RoPE | SnapKV (question hidden) | 10% | 6 | -0.192 | [-0.480, 0.043] | 4 / 0 / 2 |
| hotpotqa | SVDD post-RoPE | Initial + recent | 10% | 6 | 0.009 | [-0.023, 0.048] | 3 / 0 / 3 |
| hotpotqa | SVDD post-RoPE | SVDD pre-RoPE | 10% | 6 | 0.041 | [0.007, 0.083] | 4 / 1 / 1 |
| hotpotqa | SVDD post-RoPE | Full cache | 20% | 6 | -0.075 | [-0.156, -0.002] | 1 / 2 / 3 |
| hotpotqa | SVDD post-RoPE | KeyDiff | 20% | 6 | 0.028 | [-0.025, 0.106] | 3 / 2 / 1 |
| hotpotqa | SVDD post-RoPE | K-norm | 20% | 6 | 0.219 | [0.010, 0.536] | 3 / 2 / 1 |
| hotpotqa | SVDD post-RoPE | Leverage only | 20% | 6 | 0.044 | [0.000, 0.098] | 2 / 4 / 0 |
| hotpotqa | SVDD post-RoPE | Random | 20% | 6 | 0.168 | [0.011, 0.423] | 4 / 2 / 0 |
| hotpotqa | SVDD post-RoPE | SnapKV (question hidden) | 20% | 6 | 0.032 | [-0.015, 0.114] | 1 / 2 / 3 |
| hotpotqa | SVDD post-RoPE | Initial + recent | 20% | 6 | 0.117 | [-0.062, 0.394] | 2 / 1 / 3 |
| hotpotqa | SVDD post-RoPE | SVDD pre-RoPE | 20% | 6 | -0.017 | [-0.065, 0.016] | 2 / 2 / 2 |
| hotpotqa | SVDD post-RoPE | Full cache | 50% | 6 | -0.013 | [-0.040, 0.006] | 1 / 3 / 2 |
| hotpotqa | SVDD post-RoPE | KeyDiff | 50% | 6 | 0.049 | [-0.026, 0.142] | 3 / 1 / 2 |
| hotpotqa | SVDD post-RoPE | K-norm | 50% | 6 | 0.230 | [0.033, 0.477] | 3 / 2 / 1 |
| hotpotqa | SVDD post-RoPE | Leverage only | 50% | 6 | 0.003 | [-0.009, 0.015] | 2 / 3 / 1 |
| hotpotqa | SVDD post-RoPE | Random | 50% | 6 | 0.160 | [-0.021, 0.425] | 3 / 2 / 1 |
| hotpotqa | SVDD post-RoPE | SnapKV (question hidden) | 50% | 6 | -0.015 | [-0.158, 0.109] | 2 / 1 / 3 |
| hotpotqa | SVDD post-RoPE | Initial + recent | 50% | 6 | 0.121 | [-0.035, 0.402] | 2 / 2 / 2 |
| hotpotqa | SVDD post-RoPE | SVDD pre-RoPE | 50% | 6 | 0.041 | [0.000, 0.119] | 2 / 4 / 0 |
| hotpotqa | SVDD pre-RoPE | Full cache | 10% | 6 | -0.288 | [-0.571, -0.049] | 1 / 1 / 4 |
| hotpotqa | SVDD pre-RoPE | KeyDiff | 10% | 6 | -0.169 | [-0.481, 0.001] | 1 / 1 / 4 |
| hotpotqa | SVDD pre-RoPE | K-norm | 10% | 6 | -0.030 | [-0.078, 0.007] | 1 / 2 / 3 |
| hotpotqa | SVDD pre-RoPE | Leverage only | 10% | 6 | -0.224 | [-0.528, -0.003] | 1 / 2 / 3 |
| hotpotqa | SVDD pre-RoPE | Random | 10% | 6 | -0.034 | [-0.080, -0.001] | 1 / 2 / 3 |
| hotpotqa | SVDD pre-RoPE | SnapKV (question hidden) | 10% | 6 | -0.233 | [-0.550, 0.010] | 2 / 2 / 2 |
| hotpotqa | SVDD pre-RoPE | Initial + recent | 10% | 6 | -0.032 | [-0.083, 0.010] | 2 / 1 / 3 |
| hotpotqa | SVDD pre-RoPE | SVDD post-RoPE | 10% | 6 | -0.041 | [-0.083, -0.007] | 1 / 1 / 4 |
| hotpotqa | SVDD pre-RoPE | Full cache | 20% | 6 | -0.057 | [-0.148, 0.000] | 1 / 2 / 3 |
| hotpotqa | SVDD pre-RoPE | KeyDiff | 20% | 6 | 0.045 | [-0.022, 0.162] | 2 / 2 / 2 |
| hotpotqa | SVDD pre-RoPE | K-norm | 20% | 6 | 0.236 | [0.010, 0.560] | 4 / 1 / 1 |
| hotpotqa | SVDD pre-RoPE | Leverage only | 20% | 6 | 0.061 | [0.003, 0.153] | 3 / 2 / 1 |
| hotpotqa | SVDD pre-RoPE | Random | 20% | 6 | 0.185 | [0.010, 0.445] | 5 / 1 / 0 |
| hotpotqa | SVDD pre-RoPE | SnapKV (question hidden) | 20% | 6 | 0.049 | [-0.026, 0.177] | 1 / 2 / 3 |
| hotpotqa | SVDD pre-RoPE | Initial + recent | 20% | 6 | 0.135 | [-0.013, 0.403] | 2 / 1 / 3 |
| hotpotqa | SVDD pre-RoPE | SVDD post-RoPE | 20% | 6 | 0.017 | [-0.014, 0.061] | 2 / 2 / 2 |
| hotpotqa | SVDD pre-RoPE | Full cache | 50% | 6 | -0.054 | [-0.133, -0.001] | 1 / 2 / 3 |
| hotpotqa | SVDD pre-RoPE | KeyDiff | 50% | 6 | 0.008 | [-0.032, 0.065] | 2 / 1 / 3 |
| hotpotqa | SVDD pre-RoPE | K-norm | 50% | 6 | 0.189 | [-0.010, 0.457] | 3 / 2 / 1 |
| hotpotqa | SVDD pre-RoPE | Leverage only | 50% | 6 | -0.038 | [-0.120, 0.009] | 2 / 2 / 2 |
| hotpotqa | SVDD pre-RoPE | Random | 50% | 6 | 0.119 | [-0.034, 0.397] | 2 / 2 / 2 |
| hotpotqa | SVDD pre-RoPE | SnapKV (question hidden) | 50% | 6 | -0.056 | [-0.162, 0.009] | 1 / 2 / 3 |
| hotpotqa | SVDD pre-RoPE | Initial + recent | 50% | 6 | 0.080 | [-0.122, 0.393] | 2 / 1 / 3 |
| hotpotqa | SVDD pre-RoPE | SVDD post-RoPE | 50% | 6 | -0.041 | [-0.119, 0.000] | 0 / 4 / 2 |
| niah_multi | SVDD post-RoPE | Full cache | 10% | 18 | -0.861 | [-0.972, -0.750] | 0 / 0 / 18 |
| niah_multi | SVDD post-RoPE | KeyDiff | 10% | 18 | -0.833 | [-0.944, -0.722] | 0 / 0 / 18 |
| niah_multi | SVDD post-RoPE | K-norm | 10% | 18 | 0.139 | [0.028, 0.250] | 5 / 13 / 0 |
| niah_multi | SVDD post-RoPE | Leverage only | 10% | 18 | -0.806 | [-0.917, -0.694] | 0 / 0 / 18 |
| niah_multi | SVDD post-RoPE | Random | 10% | 18 | 0.139 | [0.056, 0.250] | 5 / 13 / 0 |
| niah_multi | SVDD post-RoPE | SnapKV (question hidden) | 10% | 18 | 0.139 | [0.056, 0.250] | 5 / 13 / 0 |
| niah_multi | SVDD post-RoPE | Initial + recent | 10% | 18 | 0.139 | [0.056, 0.250] | 5 / 13 / 0 |
| niah_multi | SVDD post-RoPE | SVDD pre-RoPE | 10% | 18 | 0.000 | [-0.083, 0.083] | 1 / 16 / 1 |
| niah_multi | SVDD post-RoPE | Full cache | 20% | 18 | -0.583 | [-0.722, -0.444] | 0 / 2 / 16 |
| niah_multi | SVDD post-RoPE | KeyDiff | 20% | 18 | -0.583 | [-0.722, -0.444] | 0 / 2 / 16 |
| niah_multi | SVDD post-RoPE | K-norm | 20% | 18 | 0.417 | [0.278, 0.556] | 13 / 5 / 0 |
| niah_multi | SVDD post-RoPE | Leverage only | 20% | 18 | -0.583 | [-0.722, -0.444] | 0 / 2 / 16 |
| niah_multi | SVDD post-RoPE | Random | 20% | 18 | 0.417 | [0.278, 0.556] | 13 / 5 / 0 |
| niah_multi | SVDD post-RoPE | SnapKV (question hidden) | 20% | 18 | 0.306 | [0.139, 0.472] | 10 / 7 / 1 |
| niah_multi | SVDD post-RoPE | Initial + recent | 20% | 18 | 0.083 | [-0.111, 0.306] | 7 / 6 / 5 |
| niah_multi | SVDD post-RoPE | SVDD pre-RoPE | 20% | 18 | 0.056 | [-0.083, 0.194] | 5 / 10 / 3 |
| niah_multi | SVDD post-RoPE | Full cache | 50% | 18 | -0.333 | [-0.417, -0.222] | 0 / 6 / 12 |
| niah_multi | SVDD post-RoPE | KeyDiff | 50% | 18 | -0.333 | [-0.444, -0.222] | 0 / 6 / 12 |
| niah_multi | SVDD post-RoPE | K-norm | 50% | 18 | 0.611 | [0.500, 0.722] | 17 / 1 / 0 |
| niah_multi | SVDD post-RoPE | Leverage only | 50% | 18 | -0.333 | [-0.444, -0.222] | 0 / 6 / 12 |
| niah_multi | SVDD post-RoPE | Random | 50% | 18 | 0.500 | [0.333, 0.667] | 13 / 5 / 0 |
| niah_multi | SVDD post-RoPE | SnapKV (question hidden) | 50% | 18 | 0.389 | [0.194, 0.583] | 12 / 4 / 2 |
| niah_multi | SVDD post-RoPE | Initial + recent | 50% | 18 | 0.167 | [0.083, 0.278] | 6 / 12 / 0 |
| niah_multi | SVDD post-RoPE | SVDD pre-RoPE | 50% | 18 | 0.056 | [-0.056, 0.167] | 3 / 14 / 1 |
| niah_multi | SVDD pre-RoPE | Full cache | 10% | 18 | -0.861 | [-0.944, -0.750] | 0 / 0 / 18 |
| niah_multi | SVDD pre-RoPE | KeyDiff | 10% | 18 | -0.833 | [-0.944, -0.722] | 0 / 0 / 18 |
| niah_multi | SVDD pre-RoPE | K-norm | 10% | 18 | 0.139 | [0.028, 0.250] | 5 / 13 / 0 |
| niah_multi | SVDD pre-RoPE | Leverage only | 10% | 18 | -0.806 | [-0.917, -0.694] | 0 / 0 / 18 |
| niah_multi | SVDD pre-RoPE | Random | 10% | 18 | 0.139 | [0.056, 0.250] | 5 / 13 / 0 |
| niah_multi | SVDD pre-RoPE | SnapKV (question hidden) | 10% | 18 | 0.139 | [0.056, 0.250] | 5 / 13 / 0 |
| niah_multi | SVDD pre-RoPE | Initial + recent | 10% | 18 | 0.139 | [0.056, 0.250] | 5 / 13 / 0 |
| niah_multi | SVDD pre-RoPE | SVDD post-RoPE | 10% | 18 | 0.000 | [-0.083, 0.083] | 1 / 16 / 1 |
| niah_multi | SVDD pre-RoPE | Full cache | 20% | 18 | -0.639 | [-0.778, -0.472] | 0 / 2 / 16 |
| niah_multi | SVDD pre-RoPE | KeyDiff | 20% | 18 | -0.639 | [-0.778, -0.500] | 0 / 2 / 16 |
| niah_multi | SVDD pre-RoPE | K-norm | 20% | 18 | 0.361 | [0.222, 0.500] | 11 / 7 / 0 |
| niah_multi | SVDD pre-RoPE | Leverage only | 20% | 18 | -0.639 | [-0.806, -0.472] | 0 / 2 / 16 |
| niah_multi | SVDD pre-RoPE | Random | 20% | 18 | 0.361 | [0.222, 0.500] | 11 / 7 / 0 |
| niah_multi | SVDD pre-RoPE | SnapKV (question hidden) | 20% | 18 | 0.250 | [0.056, 0.444] | 9 / 7 / 2 |
| niah_multi | SVDD pre-RoPE | Initial + recent | 20% | 18 | 0.028 | [-0.167, 0.250] | 6 / 6 / 6 |
| niah_multi | SVDD pre-RoPE | SVDD post-RoPE | 20% | 18 | -0.056 | [-0.194, 0.111] | 3 / 10 / 5 |
| niah_multi | SVDD pre-RoPE | Full cache | 50% | 18 | -0.389 | [-0.472, -0.278] | 0 / 4 / 14 |
| niah_multi | SVDD pre-RoPE | KeyDiff | 50% | 18 | -0.389 | [-0.472, -0.306] | 0 / 4 / 14 |
| niah_multi | SVDD pre-RoPE | K-norm | 50% | 18 | 0.556 | [0.444, 0.667] | 17 / 1 / 0 |
| niah_multi | SVDD pre-RoPE | Leverage only | 50% | 18 | -0.389 | [-0.472, -0.278] | 0 / 4 / 14 |
| niah_multi | SVDD pre-RoPE | Random | 50% | 18 | 0.444 | [0.306, 0.583] | 13 / 5 / 0 |
| niah_multi | SVDD pre-RoPE | SnapKV (question hidden) | 50% | 18 | 0.333 | [0.139, 0.528] | 11 / 5 / 2 |
| niah_multi | SVDD pre-RoPE | Initial + recent | 50% | 18 | 0.111 | [0.028, 0.222] | 4 / 14 / 0 |
| niah_multi | SVDD pre-RoPE | SVDD post-RoPE | 50% | 18 | -0.056 | [-0.167, 0.056] | 1 / 14 / 3 |
| niah_single | SVDD post-RoPE | Full cache | 10% | 18 | -1.000 | [-1.000, -1.000] | 0 / 0 / 18 |
| niah_single | SVDD post-RoPE | KeyDiff | 10% | 18 | -1.000 | [-1.000, -1.000] | 0 / 0 / 18 |
| niah_single | SVDD post-RoPE | K-norm | 10% | 18 | 0.000 | [0.000, 0.000] | 0 / 18 / 0 |
| niah_single | SVDD post-RoPE | Leverage only | 10% | 18 | -1.000 | [-1.000, -1.000] | 0 / 0 / 18 |
| niah_single | SVDD post-RoPE | Random | 10% | 18 | 0.000 | [0.000, 0.000] | 0 / 18 / 0 |
| niah_single | SVDD post-RoPE | SnapKV (question hidden) | 10% | 18 | 0.000 | [0.000, 0.000] | 0 / 18 / 0 |
| niah_single | SVDD post-RoPE | Initial + recent | 10% | 18 | 0.000 | [0.000, 0.000] | 0 / 18 / 0 |
| niah_single | SVDD post-RoPE | SVDD pre-RoPE | 10% | 18 | -0.056 | [-0.167, 0.000] | 0 / 17 / 1 |
| niah_single | SVDD post-RoPE | Full cache | 20% | 18 | -0.778 | [-0.944, -0.556] | 0 / 4 / 14 |
| niah_single | SVDD post-RoPE | KeyDiff | 20% | 18 | -0.778 | [-0.944, -0.556] | 0 / 4 / 14 |
| niah_single | SVDD post-RoPE | K-norm | 20% | 18 | 0.222 | [0.056, 0.444] | 4 / 14 / 0 |
| niah_single | SVDD post-RoPE | Leverage only | 20% | 18 | -0.778 | [-0.944, -0.556] | 0 / 4 / 14 |
| niah_single | SVDD post-RoPE | Random | 20% | 18 | 0.222 | [0.056, 0.444] | 4 / 14 / 0 |
| niah_single | SVDD post-RoPE | SnapKV (question hidden) | 20% | 18 | 0.167 | [0.000, 0.333] | 3 / 15 / 0 |
| niah_single | SVDD post-RoPE | Initial + recent | 20% | 18 | -0.111 | [-0.333, 0.167] | 2 / 12 / 4 |
| niah_single | SVDD post-RoPE | SVDD pre-RoPE | 20% | 18 | 0.000 | [-0.278, 0.278] | 3 / 12 / 3 |
| niah_single | SVDD post-RoPE | Full cache | 50% | 18 | -0.500 | [-0.722, -0.278] | 0 / 9 / 9 |
| niah_single | SVDD post-RoPE | KeyDiff | 50% | 18 | -0.500 | [-0.722, -0.278] | 0 / 9 / 9 |
| niah_single | SVDD post-RoPE | K-norm | 50% | 18 | 0.444 | [0.222, 0.667] | 8 / 10 / 0 |
| niah_single | SVDD post-RoPE | Leverage only | 50% | 18 | -0.500 | [-0.722, -0.278] | 0 / 9 / 9 |
| niah_single | SVDD post-RoPE | Random | 50% | 18 | 0.333 | [0.000, 0.611] | 8 / 8 / 2 |
| niah_single | SVDD post-RoPE | SnapKV (question hidden) | 50% | 18 | 0.056 | [-0.278, 0.444] | 6 / 7 / 5 |
| niah_single | SVDD post-RoPE | Initial + recent | 50% | 18 | -0.167 | [-0.556, 0.278] | 6 / 3 / 9 |
| niah_single | SVDD post-RoPE | SVDD pre-RoPE | 50% | 18 | 0.056 | [-0.111, 0.222] | 2 / 15 / 1 |
| niah_single | SVDD pre-RoPE | Full cache | 10% | 18 | -0.944 | [-1.000, -0.833] | 0 / 1 / 17 |
| niah_single | SVDD pre-RoPE | KeyDiff | 10% | 18 | -0.944 | [-1.000, -0.833] | 0 / 1 / 17 |
| niah_single | SVDD pre-RoPE | K-norm | 10% | 18 | 0.056 | [0.000, 0.167] | 1 / 17 / 0 |
| niah_single | SVDD pre-RoPE | Leverage only | 10% | 18 | -0.944 | [-1.000, -0.833] | 0 / 1 / 17 |
| niah_single | SVDD pre-RoPE | Random | 10% | 18 | 0.056 | [0.000, 0.167] | 1 / 17 / 0 |
| niah_single | SVDD pre-RoPE | SnapKV (question hidden) | 10% | 18 | 0.056 | [0.000, 0.167] | 1 / 17 / 0 |
| niah_single | SVDD pre-RoPE | Initial + recent | 10% | 18 | 0.056 | [0.000, 0.167] | 1 / 17 / 0 |
| niah_single | SVDD pre-RoPE | SVDD post-RoPE | 10% | 18 | 0.056 | [0.000, 0.167] | 1 / 17 / 0 |
| niah_single | SVDD pre-RoPE | Full cache | 20% | 18 | -0.778 | [-0.944, -0.611] | 0 / 4 / 14 |
| niah_single | SVDD pre-RoPE | KeyDiff | 20% | 18 | -0.778 | [-0.944, -0.556] | 0 / 4 / 14 |
| niah_single | SVDD pre-RoPE | K-norm | 20% | 18 | 0.222 | [0.056, 0.444] | 4 / 14 / 0 |
| niah_single | SVDD pre-RoPE | Leverage only | 20% | 18 | -0.778 | [-0.944, -0.556] | 0 / 4 / 14 |
| niah_single | SVDD pre-RoPE | Random | 20% | 18 | 0.222 | [0.056, 0.444] | 4 / 14 / 0 |
| niah_single | SVDD pre-RoPE | SnapKV (question hidden) | 20% | 18 | 0.167 | [-0.056, 0.389] | 4 / 13 / 1 |
| niah_single | SVDD pre-RoPE | Initial + recent | 20% | 18 | -0.111 | [-0.389, 0.167] | 2 / 12 / 4 |
| niah_single | SVDD pre-RoPE | SVDD post-RoPE | 20% | 18 | 0.000 | [-0.278, 0.278] | 3 / 12 / 3 |
| niah_single | SVDD pre-RoPE | Full cache | 50% | 18 | -0.556 | [-0.778, -0.333] | 0 / 8 / 10 |
| niah_single | SVDD pre-RoPE | KeyDiff | 50% | 18 | -0.556 | [-0.778, -0.333] | 0 / 8 / 10 |
| niah_single | SVDD pre-RoPE | K-norm | 50% | 18 | 0.389 | [0.111, 0.667] | 8 / 9 / 1 |
| niah_single | SVDD pre-RoPE | Leverage only | 50% | 18 | -0.556 | [-0.778, -0.333] | 0 / 8 / 10 |
| niah_single | SVDD pre-RoPE | Random | 50% | 18 | 0.278 | [0.000, 0.611] | 7 / 9 / 2 |
| niah_single | SVDD pre-RoPE | SnapKV (question hidden) | 50% | 18 | 0.000 | [-0.389, 0.389] | 7 / 4 / 7 |
| niah_single | SVDD pre-RoPE | Initial + recent | 50% | 18 | -0.222 | [-0.611, 0.222] | 6 / 2 / 10 |
| niah_single | SVDD pre-RoPE | SVDD post-RoPE | 50% | 18 | -0.056 | [-0.222, 0.111] | 1 / 15 / 2 |
| passage_retrieval_en | SVDD post-RoPE | Full cache | 10% | 6 | -0.833 | [-1.000, -0.500] | 0 / 1 / 5 |
| passage_retrieval_en | SVDD post-RoPE | KeyDiff | 10% | 6 | 0.000 | [-0.500, 0.500] | 1 / 4 / 1 |
| passage_retrieval_en | SVDD post-RoPE | K-norm | 10% | 6 | 0.000 | [-0.500, 0.500] | 1 / 4 / 1 |
| passage_retrieval_en | SVDD post-RoPE | Leverage only | 10% | 6 | -0.500 | [-0.833, -0.167] | 0 / 3 / 3 |
| passage_retrieval_en | SVDD post-RoPE | Random | 10% | 6 | 0.000 | [-0.500, 0.500] | 1 / 4 / 1 |
| passage_retrieval_en | SVDD post-RoPE | SnapKV (question hidden) | 10% | 6 | -0.333 | [-0.667, 0.000] | 0 / 4 / 2 |
| passage_retrieval_en | SVDD post-RoPE | Initial + recent | 10% | 6 | 0.167 | [0.000, 0.500] | 1 / 5 / 0 |
| passage_retrieval_en | SVDD post-RoPE | SVDD pre-RoPE | 10% | 6 | -0.333 | [-0.667, 0.000] | 0 / 4 / 2 |
| passage_retrieval_en | SVDD post-RoPE | Full cache | 20% | 6 | -0.333 | [-0.667, 0.000] | 0 / 4 / 2 |
| passage_retrieval_en | SVDD post-RoPE | KeyDiff | 20% | 6 | -0.167 | [-0.500, 0.000] | 0 / 5 / 1 |
| passage_retrieval_en | SVDD post-RoPE | K-norm | 20% | 6 | 0.667 | [0.333, 1.000] | 4 / 2 / 0 |
| passage_retrieval_en | SVDD post-RoPE | Leverage only | 20% | 6 | -0.167 | [-0.500, 0.000] | 0 / 5 / 1 |
| passage_retrieval_en | SVDD post-RoPE | Random | 20% | 6 | 0.500 | [0.167, 0.833] | 3 / 3 / 0 |
| passage_retrieval_en | SVDD post-RoPE | SnapKV (question hidden) | 20% | 6 | -0.167 | [-0.500, 0.000] | 0 / 5 / 1 |
| passage_retrieval_en | SVDD post-RoPE | Initial + recent | 20% | 6 | 0.667 | [0.333, 1.000] | 4 / 2 / 0 |
| passage_retrieval_en | SVDD post-RoPE | SVDD pre-RoPE | 20% | 6 | -0.167 | [-0.500, 0.000] | 0 / 5 / 1 |
| passage_retrieval_en | SVDD post-RoPE | Full cache | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD post-RoPE | KeyDiff | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD post-RoPE | K-norm | 50% | 6 | 0.500 | [0.167, 0.833] | 3 / 3 / 0 |
| passage_retrieval_en | SVDD post-RoPE | Leverage only | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD post-RoPE | Random | 50% | 6 | 0.167 | [0.000, 0.500] | 1 / 5 / 0 |
| passage_retrieval_en | SVDD post-RoPE | SnapKV (question hidden) | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD post-RoPE | Initial + recent | 50% | 6 | 1.000 | [1.000, 1.000] | 6 / 0 / 0 |
| passage_retrieval_en | SVDD post-RoPE | SVDD pre-RoPE | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | Full cache | 10% | 6 | -0.500 | [-0.833, -0.167] | 0 / 3 / 3 |
| passage_retrieval_en | SVDD pre-RoPE | KeyDiff | 10% | 6 | 0.333 | [0.000, 0.667] | 2 / 4 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | K-norm | 10% | 6 | 0.333 | [0.000, 0.667] | 2 / 4 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | Leverage only | 10% | 6 | -0.167 | [-0.667, 0.333] | 1 / 3 / 2 |
| passage_retrieval_en | SVDD pre-RoPE | Random | 10% | 6 | 0.333 | [0.000, 0.667] | 2 / 4 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | SnapKV (question hidden) | 10% | 6 | 0.000 | [-0.500, 0.500] | 1 / 4 / 1 |
| passage_retrieval_en | SVDD pre-RoPE | Initial + recent | 10% | 6 | 0.500 | [0.167, 0.833] | 3 / 3 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | SVDD post-RoPE | 10% | 6 | 0.333 | [0.000, 0.667] | 2 / 4 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | Full cache | 20% | 6 | -0.167 | [-0.500, 0.000] | 0 / 5 / 1 |
| passage_retrieval_en | SVDD pre-RoPE | KeyDiff | 20% | 6 | 0.000 | [-0.500, 0.500] | 1 / 4 / 1 |
| passage_retrieval_en | SVDD pre-RoPE | K-norm | 20% | 6 | 0.833 | [0.500, 1.000] | 5 / 1 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | Leverage only | 20% | 6 | 0.000 | [-0.500, 0.500] | 1 / 4 / 1 |
| passage_retrieval_en | SVDD pre-RoPE | Random | 20% | 6 | 0.667 | [0.333, 1.000] | 4 / 2 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | SnapKV (question hidden) | 20% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | Initial + recent | 20% | 6 | 0.833 | [0.500, 1.000] | 5 / 1 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | SVDD post-RoPE | 20% | 6 | 0.167 | [0.000, 0.500] | 1 / 5 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | Full cache | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | KeyDiff | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | K-norm | 50% | 6 | 0.500 | [0.167, 0.833] | 3 / 3 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | Leverage only | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | Random | 50% | 6 | 0.167 | [0.000, 0.500] | 1 / 5 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | SnapKV (question hidden) | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | Initial + recent | 50% | 6 | 1.000 | [1.000, 1.000] | 6 / 0 / 0 |
| passage_retrieval_en | SVDD pre-RoPE | SVDD post-RoPE | 50% | 6 | 0.000 | [0.000, 0.000] | 0 / 6 / 0 |
| qasper | SVDD post-RoPE | Full cache | 10% | 6 | -0.105 | [-0.163, -0.036] | 1 / 0 / 5 |
| qasper | SVDD post-RoPE | KeyDiff | 10% | 6 | 0.030 | [-0.020, 0.073] | 4 / 1 / 1 |
| qasper | SVDD post-RoPE | K-norm | 10% | 6 | 0.032 | [-0.003, 0.072] | 3 / 1 / 2 |
| qasper | SVDD post-RoPE | Leverage only | 10% | 6 | 0.017 | [-0.016, 0.053] | 3 / 1 / 2 |
| qasper | SVDD post-RoPE | Random | 10% | 6 | -0.001 | [-0.036, 0.026] | 4 / 1 / 1 |
| qasper | SVDD post-RoPE | SnapKV (question hidden) | 10% | 6 | 0.038 | [-0.003, 0.074] | 4 / 1 / 1 |
| qasper | SVDD post-RoPE | Initial + recent | 10% | 6 | -0.018 | [-0.099, 0.046] | 3 / 1 / 2 |
| qasper | SVDD post-RoPE | SVDD pre-RoPE | 10% | 6 | -0.008 | [-0.047, 0.021] | 3 / 1 / 2 |
| qasper | SVDD post-RoPE | Full cache | 20% | 6 | 0.034 | [-0.094, 0.163] | 4 / 0 / 2 |
| qasper | SVDD post-RoPE | KeyDiff | 20% | 6 | 0.107 | [-0.056, 0.254] | 4 / 1 / 1 |
| qasper | SVDD post-RoPE | K-norm | 20% | 6 | 0.176 | [0.048, 0.329] | 4 / 1 / 1 |
| qasper | SVDD post-RoPE | Leverage only | 20% | 6 | 0.074 | [-0.041, 0.204] | 4 / 0 / 2 |
| qasper | SVDD post-RoPE | Random | 20% | 6 | 0.081 | [-0.055, 0.224] | 3 / 1 / 2 |
| qasper | SVDD post-RoPE | SnapKV (question hidden) | 20% | 6 | 0.047 | [-0.057, 0.159] | 3 / 1 / 2 |
| qasper | SVDD post-RoPE | Initial + recent | 20% | 6 | 0.105 | [-0.020, 0.252] | 3 / 1 / 2 |
| qasper | SVDD post-RoPE | SVDD pre-RoPE | 20% | 6 | 0.084 | [0.012, 0.175] | 4 / 1 / 1 |
| qasper | SVDD post-RoPE | Full cache | 50% | 6 | -0.026 | [-0.103, 0.047] | 3 / 1 / 2 |
| qasper | SVDD post-RoPE | KeyDiff | 50% | 6 | -0.036 | [-0.126, 0.038] | 3 / 0 / 3 |
| qasper | SVDD post-RoPE | K-norm | 50% | 6 | 0.025 | [-0.006, 0.055] | 4 / 1 / 1 |
| qasper | SVDD post-RoPE | Leverage only | 50% | 6 | -0.026 | [-0.069, 0.015] | 1 / 0 / 5 |
| qasper | SVDD post-RoPE | Random | 50% | 6 | -0.042 | [-0.141, 0.054] | 3 / 0 / 3 |
| qasper | SVDD post-RoPE | SnapKV (question hidden) | 50% | 6 | -0.021 | [-0.082, 0.039] | 2 / 1 / 3 |
| qasper | SVDD post-RoPE | Initial + recent | 50% | 6 | -0.041 | [-0.136, 0.059] | 2 / 0 / 4 |
| qasper | SVDD post-RoPE | SVDD pre-RoPE | 50% | 6 | 0.004 | [-0.021, 0.031] | 2 / 2 / 2 |
| qasper | SVDD pre-RoPE | Full cache | 10% | 6 | -0.098 | [-0.169, -0.026] | 1 / 1 / 4 |
| qasper | SVDD pre-RoPE | KeyDiff | 10% | 6 | 0.038 | [-0.040, 0.119] | 4 / 1 / 1 |
| qasper | SVDD pre-RoPE | K-norm | 10% | 6 | 0.039 | [-0.014, 0.110] | 2 / 1 / 3 |
| qasper | SVDD pre-RoPE | Leverage only | 10% | 6 | 0.025 | [0.005, 0.049] | 4 / 1 / 1 |
| qasper | SVDD pre-RoPE | Random | 10% | 6 | 0.007 | [-0.042, 0.062] | 2 / 1 / 3 |
| qasper | SVDD pre-RoPE | SnapKV (question hidden) | 10% | 6 | 0.045 | [-0.009, 0.116] | 4 / 1 / 1 |
| qasper | SVDD pre-RoPE | Initial + recent | 10% | 6 | -0.010 | [-0.103, 0.088] | 2 / 1 / 3 |
| qasper | SVDD pre-RoPE | SVDD post-RoPE | 10% | 6 | 0.008 | [-0.021, 0.047] | 2 / 1 / 3 |
| qasper | SVDD pre-RoPE | Full cache | 20% | 6 | -0.050 | [-0.130, 0.028] | 2 / 0 / 4 |
| qasper | SVDD pre-RoPE | KeyDiff | 20% | 6 | 0.022 | [-0.093, 0.104] | 4 / 1 / 1 |
| qasper | SVDD pre-RoPE | K-norm | 20% | 6 | 0.091 | [0.022, 0.165] | 4 / 1 / 1 |
| qasper | SVDD pre-RoPE | Leverage only | 20% | 6 | -0.011 | [-0.075, 0.047] | 4 / 0 / 2 |
| qasper | SVDD pre-RoPE | Random | 20% | 6 | -0.003 | [-0.090, 0.063] | 3 / 1 / 2 |
| qasper | SVDD pre-RoPE | SnapKV (question hidden) | 20% | 6 | -0.038 | [-0.087, 0.005] | 1 / 1 / 4 |
| qasper | SVDD pre-RoPE | Initial + recent | 20% | 6 | 0.021 | [-0.049, 0.089] | 3 / 1 / 2 |
| qasper | SVDD pre-RoPE | SVDD post-RoPE | 20% | 6 | -0.084 | [-0.175, -0.016] | 1 / 1 / 4 |
| qasper | SVDD pre-RoPE | Full cache | 50% | 6 | -0.029 | [-0.114, 0.046] | 4 / 0 / 2 |
| qasper | SVDD pre-RoPE | KeyDiff | 50% | 6 | -0.040 | [-0.134, 0.048] | 2 / 1 / 3 |
| qasper | SVDD pre-RoPE | K-norm | 50% | 6 | 0.022 | [-0.025, 0.072] | 3 / 1 / 2 |
| qasper | SVDD pre-RoPE | Leverage only | 50% | 6 | -0.029 | [-0.068, 0.004] | 1 / 1 / 4 |
| qasper | SVDD pre-RoPE | Random | 50% | 6 | -0.046 | [-0.160, 0.060] | 2 / 0 / 4 |
| qasper | SVDD pre-RoPE | SnapKV (question hidden) | 50% | 6 | -0.025 | [-0.106, 0.050] | 2 / 1 / 3 |
| qasper | SVDD pre-RoPE | Initial + recent | 50% | 6 | -0.044 | [-0.160, 0.075] | 2 / 0 / 4 |
| qasper | SVDD pre-RoPE | SVDD post-RoPE | 50% | 6 | -0.004 | [-0.031, 0.022] | 2 / 2 / 2 |

All method–baseline comparisons, length-specific deltas, pair counts, and missing-pair diagnostics are in `paired_deltas.csv`.

## Retrieval cases solved by full cache

There are **42 retrieval cases** with full-cache score exactly 1. This subgroup includes custom NIAH and LongBench passage retrieval; partial full-cache multi-needle matches are excluded. It asks whether compression preserves success where the base model succeeds. It is a conditional, post-selection diagnostic, not a replacement for all-case accuracy. Missing full-cache rows are not treated as failures.

| Task | Method | Retention | Cases | Conditional mean score |
|---|---|---:|---:|---:|
| niah_multi | Full cache | 100% | 18 | 1.000 |
| niah_multi | KeyDiff | 10% | 18 | 0.972 |
| niah_multi | KeyDiff | 20% | 18 | 1.000 |
| niah_multi | KeyDiff | 50% | 18 | 1.000 |
| niah_multi | K-norm | 10% | 18 | 0.000 |
| niah_multi | K-norm | 20% | 18 | 0.000 |
| niah_multi | K-norm | 50% | 18 | 0.056 |
| niah_multi | Leverage only | 10% | 18 | 0.944 |
| niah_multi | Leverage only | 20% | 18 | 1.000 |
| niah_multi | Leverage only | 50% | 18 | 1.000 |
| niah_multi | Random | 10% | 18 | 0.000 |
| niah_multi | Random | 20% | 18 | 0.000 |
| niah_multi | Random | 50% | 18 | 0.167 |
| niah_multi | SnapKV (question hidden) | 10% | 18 | 0.000 |
| niah_multi | SnapKV (question hidden) | 20% | 18 | 0.111 |
| niah_multi | SnapKV (question hidden) | 50% | 18 | 0.278 |
| niah_multi | Initial + recent | 10% | 18 | 0.000 |
| niah_multi | Initial + recent | 20% | 18 | 0.333 |
| niah_multi | Initial + recent | 50% | 18 | 0.500 |
| niah_multi | SVDD post-RoPE | 10% | 18 | 0.139 |
| niah_multi | SVDD post-RoPE | 20% | 18 | 0.417 |
| niah_multi | SVDD post-RoPE | 50% | 18 | 0.667 |
| niah_multi | SVDD pre-RoPE | 10% | 18 | 0.139 |
| niah_multi | SVDD pre-RoPE | 20% | 18 | 0.361 |
| niah_multi | SVDD pre-RoPE | 50% | 18 | 0.611 |
| niah_single | Full cache | 100% | 18 | 1.000 |
| niah_single | KeyDiff | 10% | 18 | 1.000 |
| niah_single | KeyDiff | 20% | 18 | 1.000 |
| niah_single | KeyDiff | 50% | 18 | 1.000 |
| niah_single | K-norm | 10% | 18 | 0.000 |
| niah_single | K-norm | 20% | 18 | 0.000 |
| niah_single | K-norm | 50% | 18 | 0.056 |
| niah_single | Leverage only | 10% | 18 | 1.000 |
| niah_single | Leverage only | 20% | 18 | 1.000 |
| niah_single | Leverage only | 50% | 18 | 1.000 |
| niah_single | Random | 10% | 18 | 0.000 |
| niah_single | Random | 20% | 18 | 0.000 |
| niah_single | Random | 50% | 18 | 0.167 |
| niah_single | SnapKV (question hidden) | 10% | 18 | 0.000 |
| niah_single | SnapKV (question hidden) | 20% | 18 | 0.056 |
| niah_single | SnapKV (question hidden) | 50% | 18 | 0.444 |
| niah_single | Initial + recent | 10% | 18 | 0.000 |
| niah_single | Initial + recent | 20% | 18 | 0.333 |
| niah_single | Initial + recent | 50% | 18 | 0.667 |
| niah_single | SVDD post-RoPE | 10% | 18 | 0.000 |
| niah_single | SVDD post-RoPE | 20% | 18 | 0.222 |
| niah_single | SVDD post-RoPE | 50% | 18 | 0.500 |
| niah_single | SVDD pre-RoPE | 10% | 18 | 0.056 |
| niah_single | SVDD pre-RoPE | 20% | 18 | 0.222 |
| niah_single | SVDD pre-RoPE | 50% | 18 | 0.444 |
| passage_retrieval_en | Full cache | 100% | 6 | 1.000 |
| passage_retrieval_en | KeyDiff | 10% | 6 | 0.167 |
| passage_retrieval_en | KeyDiff | 20% | 6 | 0.833 |
| passage_retrieval_en | KeyDiff | 50% | 6 | 1.000 |
| passage_retrieval_en | K-norm | 10% | 6 | 0.167 |
| passage_retrieval_en | K-norm | 20% | 6 | 0.000 |
| passage_retrieval_en | K-norm | 50% | 6 | 0.500 |
| passage_retrieval_en | Leverage only | 10% | 6 | 0.667 |
| passage_retrieval_en | Leverage only | 20% | 6 | 0.833 |
| passage_retrieval_en | Leverage only | 50% | 6 | 1.000 |
| passage_retrieval_en | Random | 10% | 6 | 0.167 |
| passage_retrieval_en | Random | 20% | 6 | 0.167 |
| passage_retrieval_en | Random | 50% | 6 | 0.833 |
| passage_retrieval_en | SnapKV (question hidden) | 10% | 6 | 0.500 |
| passage_retrieval_en | SnapKV (question hidden) | 20% | 6 | 0.833 |
| passage_retrieval_en | SnapKV (question hidden) | 50% | 6 | 1.000 |
| passage_retrieval_en | Initial + recent | 10% | 6 | 0.000 |
| passage_retrieval_en | Initial + recent | 20% | 6 | 0.000 |
| passage_retrieval_en | Initial + recent | 50% | 6 | 0.000 |
| passage_retrieval_en | SVDD post-RoPE | 10% | 6 | 0.167 |
| passage_retrieval_en | SVDD post-RoPE | 20% | 6 | 0.667 |
| passage_retrieval_en | SVDD post-RoPE | 50% | 6 | 1.000 |
| passage_retrieval_en | SVDD pre-RoPE | 10% | 6 | 0.500 |
| passage_retrieval_en | SVDD pre-RoPE | 20% | 6 | 0.833 |
| passage_retrieval_en | SVDD pre-RoPE | 50% | 6 | 1.000 |

## Cost interpretation and artifacts

- **Tensor bytes** are actual compacted prefix K/V tensor payloads, before adding question and generation tokens. `kv_bytes_after_generation` records later growth. These byte counts exclude weights, selector workspaces, allocator padding, and shared full-cache copies.
- `process_rss_bytes` and `mlx_peak_bytes_shared_harness` describe a process that retains a full reference and shares allocations between variants. They are retained in CSV for transparency and **must not be described as independent per-method peak memory savings**.
- Compression overhead includes key transfer, the complete score computation, and selection/compaction. The runner computes scores once and reuses them for three retention budgets; each estimated deployment total pays the complete score cost once, without dividing it by three. Estimated total is prefill + compression overhead + question/generation time. It is an accounting estimate assembled from shared measurements, not an isolated end-to-end deployment benchmark.
- Output lengths vary. Decode tokens/s and generation time are not interchangeable, especially for very short answers. Weights can be quantized while K/V remains unquantized; this pilot is not evidence against KV quantization.
- `aggregate_by_task.csv`: per-task quality, payload sizes, timing, and diagnostic averages with measurement counts.
- `niah_by_length.csv`: custom single/multiple-needle scores split by nominal length.
- `niah_systems_by_length.csv`: equal-case-weight pooled custom NIAH cost/quality by length; inspect component tasks separately.
- `full_cache_correct_retrieval.csv`: the conditional retrieval subgroup.
- `paired_deltas.csv`: all paired deltas and case-bootstrap intervals; `missing_predictions.csv`: expected missing cells.
- `quality_by_task`, `niah_by_length`, and `niah_quality_cost` plots are exported as standalone PNG/PDF where applicable.
- `summary_manifest.json` records source hash, summarizer hash, environment, bootstrap settings, and artifact names.

No paper go/no-go verdict is generated automatically. Read raw answers, full-cache performance, selection diagnostics, and missing coverage alongside these averages.
