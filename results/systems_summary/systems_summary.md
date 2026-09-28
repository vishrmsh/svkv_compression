# Isolated systems measurements

One nominal-32k case (`niah_single_32768_s17_d0.1`; 32,512 prefix tokens), 2 fresh-process repetitions per method. Cells show **median [minimum, maximum]**. The observed ranges describe these repetitions; they are not confidence intervals.

Every arm uses reserved append buffers and 32 one-token forward steps after the same suffix, continuing after EOS. These measurements describe this Apple/MLX setup; they do not establish CUDA performance. Quality-harness concatenate timings are a separate measurement.

## Memory

Active Δ is post-release MLX active memory minus the warmed model/runtime baseline. Active total includes weights. Logical KV excludes unused reserve; capacity counts backing-array extent including that reserve. MLX active memory additionally captures allocator alignment. The MLX peak includes the full prefill and compaction overlap. Current RSS is sampled after releasing the original cache; peak RSS is the process-lifetime high-water mark, including model loading and warmup.

| Method | Active Δ MiB | Active total GiB | Logical KV MiB | KV capacity MiB | Peak MLX GiB | Current RSS GiB | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full cache | 1781.50 [1781.50, 1781.50] | 5.73 [5.73, 5.73] | 1778.00 [1778.00, 1778.00] | 1781.06 [1781.06, 1781.06] | 6.11 [6.11, 6.11] | 4.27 [4.27, 4.27] | 4.30 [4.30, 4.30] |
| SVDD post-RoPE | 358.75 [358.75, 358.75] | 4.34 [4.34, 4.34] | 355.58 [355.58, 355.58] | 358.64 [358.64, 358.64] | 6.43 [6.43, 6.43] | 4.80 [4.80, 4.81] | 4.80 [4.80, 4.81] |
| SVDD pre-RoPE | 358.75 [358.75, 358.75] | 4.34 [4.34, 4.34] | 355.58 [355.58, 355.58] | 358.64 [358.64, 358.64] | 6.43 [6.43, 6.43] | 5.03 [5.03, 5.04] | 5.03 [5.03, 5.04] |
| KeyDiff | 358.75 [358.75, 358.75] | 4.34 [4.34, 4.34] | 355.58 [355.58, 355.58] | 358.64 [358.64, 358.64] | 6.43 [6.43, 6.43] | 4.84 [4.84, 4.85] | 4.84 [4.84, 4.85] |
| Leverage only | 358.75 [358.75, 358.75] | 4.34 [4.34, 4.34] | 355.58 [355.58, 355.58] | 358.64 [358.64, 358.64] | 6.43 [6.43, 6.43] | 4.72 [4.72, 4.72] | 4.72 [4.72, 4.72] |
| SnapKV-style | 358.75 [358.75, 358.75] | 4.34 [4.34, 4.34] | 355.58 [355.58, 355.58] | 358.64 [358.64, 358.64] | 6.44 [6.44, 6.44] | 4.29 [4.29, 4.29] | 4.30 [4.30, 4.30] |
| Initial + recent | 358.75 [358.75, 358.75] | 4.34 [4.34, 4.34] | 355.58 [355.58, 355.58] | 358.64 [358.64, 358.64] | 6.43 [6.43, 6.43] | 4.27 [4.27, 4.28] | 4.30 [4.30, 4.30] |

## Time

Selection includes key export, scoring, ranking/gather, and allocation of the selected cache's append reserve. Prefill includes allocation of the original reserved cache. Total sums prefill, selection, suffix TTFT, and the 32 decode forwards; model loading and warmup are excluded. Decode tokens/s counts one-token forwards, rather than an EOS-terminated answer length.

| Method | Prefill s | Selection s | Suffix TTFT s | Decode tokens/s | Total s |
|---|---:|---:|---:|---:|---:|
| Full cache | 29.521 [29.480, 29.561] | 0.000 [0.000, 0.000] | 0.168 [0.168, 0.168] | 71.133 [70.892, 71.373] | 30.138 [30.099, 30.177] |
| SVDD post-RoPE | 29.575 [29.572, 29.579] | 9.898 [9.782, 10.014] | 0.069 [0.068, 0.069] | 101.820 [101.381, 102.259] | 39.856 [39.745, 39.967] |
| SVDD pre-RoPE | 29.456 [29.452, 29.461] | 10.298 [10.220, 10.376] | 0.069 [0.069, 0.070] | 101.243 [101.125, 101.361] | 40.139 [40.056, 40.222] |
| KeyDiff | 29.575 [29.501, 29.649] | 1.329 [1.324, 1.334] | 0.068 [0.065, 0.071] | 102.105 [101.643, 102.567] | 31.286 [31.208, 31.363] |
| Leverage only | 29.551 [29.462, 29.640] | 1.932 [1.926, 1.939] | 0.072 [0.071, 0.072] | 101.256 [101.124, 101.387] | 31.871 [31.790, 31.952] |
| SnapKV-style | 29.539 [29.471, 29.608] | 0.418 [0.417, 0.420] | 0.058 [0.058, 0.058] | 101.444 [100.938, 101.950] | 30.331 [30.266, 30.396] |
| Initial + recent | 29.493 [29.454, 29.532] | 0.033 [0.033, 0.033] | 0.058 [0.058, 0.058] | 101.497 [101.288, 101.707] | 29.900 [29.860, 29.939] |

Two repetitions on one context give weak precision and no estimate of workload generalization. Full-context prefill remains part of every arm's peak-memory cost. `systems_summary.csv` contains median/min/max for every recorded field; `systems_runs.csv` preserves each repetition separately.
