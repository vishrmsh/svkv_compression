# Independent data and protocol audit

Audit date: 2026-09-27. This review used CPU/tokenizer inspection only. It found a synthetic prompt-separator defect; the execution agent corrected that formatting, archived the partial run, and restarted all arms. The audit did not tune selectors or metrics against observed outcomes. It checks the scope of legitimate comparisons rather than selecting a favorable configuration.

## Data integrity

The saved case file contains **54 unique cases**: 18 single-needle, 18 two-needle, and six each from LongBench HotpotQA, Qasper, and passage retrieval. All saved prefix SHA256 values match their actual token sequences. The largest corrected prefix + question suffix + generation allowance is **32,573 tokens**, below 32,768.

CPU regeneration from the local tokenizer and original archive produced **exactly identical case objects**. The saved manifest exactly matches case metadata and token lengths, and the corrected run's invocation hash matches the saved case file. Corrected case-file SHA256: `bbf5d6cae25ebd44a8bd99b6ae93dc7c92ffac76f2c4ee8194e97fcb9d32af40`.

For all 18 LongBench cases, the complete original context appears in the decoded prefix, the original question/summary appears in the suffix, and saved answers match the source archive. None of these selected question/summary strings appears literally in its source context. Each task has six distinct source contexts. The length filter and fixed shuffled order use no answer score. This supports calling the selection deterministic and independent of observed model quality.

| Task | Source row indices, in evaluation order | Prefix-token range |
|---|---|---:|
| HotpotQA | 62, 182, 186, 117, 71, 142 | 12,458–17,151 |
| Qasper | 67, 178, 15, 127, 13, 73 | 8,153–17,811 |
| Passage retrieval English | 62, 182, 186, 22, 117, 71 | 10,789–14,912 |

The source IDs are retained in [`data/case_manifest.json`](../data/case_manifest.json). These are **LongBench-derived subsets using a custom prompt**, not a reproduction of the full official benchmark protocol. The official prompts repeat some instructions near the question; the pilot uses simpler instructions and its own chat template. Pilot maximum new tokens are 128 for HotpotQA versus the official 32, 128 for Qasper in both, and 16 for passage retrieval versus the official 32. All compared methods receive the same task-specific cap. [Official prompts](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2prompt.json), [official generation limits](https://github.com/THUDM/LongBench/blob/main/LongBench/config/dataset2maxlen.json).

## Synthetic retrieval construction

Each gold seven-digit number occurs exactly once in its decoded prefix, does not occur in its question suffix, and is contained in a recorded needle span. The measured insertion depths span approximately 0.1008–0.8995 of the prefix. Two-needle cases place the second needle at a different depth; the `depth` field describes the primary insertion choice, not both needles.

The filler pool is **11,435 tokens from three Qasper documents**. It is rotated by seed and **cyclically repeated** when necessary at 16K and 32K. Consequently, these are controlled retrieval tasks with natural-prose filler, but not 36 independent natural documents. Conspicuous magic-number statements inside repeated scientific prose may favor geometric novelty selection. Any advantage here must be checked on ordinary answer-bearing content and adversarially unusual distractors before a broad retrieval claim.

A prompt formatting defect was found during the initial execution: the synthetic question followed the final filler token without an inserted newline. The partial execution was stopped and its **525 rows over 21 cases** were archived under `results/pre_separator_check`, excluded from the final analysis. The corrected generator prepends `\n\n` to each synthetic question suffix; all 36 saved synthetic suffixes were checked. Every archived prefix hash still matches its corrected counterpart, so context keys and selector inputs are unchanged. All 54 cases were rerun from scratch under the corrected prompt. This is a disclosed implementation correction, not a quality-driven parameter sweep.

## Metrics

HotpotQA and Qasper use maximum reference-answer token F1 after the same English normalization as LongBench. Passage retrieval uses the fraction of predicted numbers equal to the gold paragraph number, also consistent with LongBench on these records. One code edge differs: an empty prediction and an answer that normalizes to empty receive 1 in this pilot and 0 in the official F1 implementation. **No selected QA answer normalizes to empty**, so this does not affect this run. [Official metrics](https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py), [official task-to-metric mapping](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py).

Custom NIAH measures **per-number recall**, using digit boundaries. It is not exact whole-answer accuracy: the correct number can appear anywhere in the output, extra guesses are not penalized, and ordering/name-to-number association is not scored. A two-needle score of 0.5 means one of two requested numbers was found. Report full two-number success separately if making an exact-answer claim.

QA token F1 must not be read as exact factual accuracy. In the corrected run, the full-cache answer to `longbench_hotpotqa_62` is a verbose, 128-token response containing the gold word “November” but also an uncertain conclusion; its F1 is approximately 0.026. The full-cache answer to `longbench_hotpotqa_182` is “East Carolina University” against “Arizona State University”, giving F1 0.333 solely from the shared word “University”. The former demonstrates sensitivity to verbosity and output-format compliance; the latter demonstrates overlap credit for a factually wrong entity. Neither example establishes that changing the metric or giving more generation tokens would recover a correct answer. The prompt, cap, and primary metric remain fixed for every arm.

The final corrected execution contains **1,350 unique rows: exactly 25 arms for every one of the 54 saved cases**. All saved task labels, prefix hashes, and reference answers match the case file. Independent rescoring of every raw prediction reproduces every recorded score, with no discrepancy. Prediction-file SHA256: `d8127ad80e5d55489bd984f739aca79c3de2f06be36e8cc83974e726a05f415b`. Only this corrected execution enters the summary; the earlier partial run is excluded.

## Final raw-output and generation-cap checks

| Task | Full-cache mean score | Full-cache rows scoring exactly 1 | Full-cache cap hits | All-arm cap hits | New-token cap |
|---|---:|---:|---:|---:|---:|
| NIAH single | 1.000000 | 18 / 18 | 0 / 18 | 75 / 450 | 32 |
| NIAH two-needle | 1.000000 | 18 / 18 | 0 / 18 | 5 / 450 | 32 |
| Passage retrieval | 1.000000 | 6 / 6 | 0 / 6 | 43 / 150 | 16 |
| HotpotQA | 0.312826 token F1 | 1 / 6 | 2 / 6 | 18 / 150 | 128 |
| Qasper | 0.189872 token F1 | 0 / 6 | 2 / 6 | 33 / 150 | 128 |

“Scoring exactly 1” uses each task's existing metric, not a new universal exact-answer metric. In particular, NIAH recall allows extra guesses; token F1 does not judge semantic equivalence; paragraph retrieval does not require an exact output string. All-arm cap counts include correlated outputs for the same contexts and are descriptive counts, not independent trials. Reaching the generation cap suggests possible truncation but does not prove that a longer output would have recovered the answer. No continuation or alternative cap was tried.

The full model solves all **42 retrieval cases** under their fixed metrics, so the full-cache-correct retrieval subgroup is the entire retrieval set here. On two-needle NIAH, neither SVDD variant ever reaches the 32-token cap, yet their mean recall is well below the geometric rivals. Those failures cannot be explained solely by the generation cap. On single-needle NIAH, post-RoPE SVDD reaches the cap on 6 of its 54 arms and pre-RoPE SVDD on 7 of 54; KeyDiff and leverage have zero cap hits across both NIAH tasks.

Raw-output spot checks cover the first saved case of every task, with full cache and the 20% post-/pre-RoPE SVDD and KeyDiff arms:

- `niah_single_8192_s17_d0.1`: full, post-RoPE SVDD, and KeyDiff output the gold `7587050`; pre-RoPE SVDD outputs `7057050777` and correctly scores zero.
- `niah_multi_8192_s17_d0.1`: full, post-RoPE SVDD, and KeyDiff output both gold numbers `6517512` and `3307403`; pre-RoPE SVDD outputs `6517512` and `3307434`, correctly scoring 0.5.
- `longbench_passage_retrieval_en_62`: all four inspected arms output `Paragraph 1`, matching the reference.
- `longbench_hotpotqa_62`: the full-cache verbosity example above was checked against the raw row; the 20% post-RoPE SVDD answer instead says the end date is absent and scores zero.
- `longbench_qasper_67`: the reference is “Logistic Regression, neural networks”. Full cache names logistic regression and a deep-learning model with additional prose (F1 0.200); post-RoPE SVDD names logistic regression and recurrent neural networks with additional prose (F1 0.421); KeyDiff says the methods are unspecified (F1 0). This illustrates why a local F1 improvement should not be generalized into a robust factual-quality claim.

The full-cache QA ceilings, tiny six-case task samples, verbose or capped responses, and reference sensitivity limit the natural-QA conclusions. Their fixed-metric differences are valid observations of this protocol, but they do not establish exact factual accuracy or preservation of general question answering. The clearest retrieval evidence is the full-capable retrieval comparison against the close geometric rivals, interpreted within its disclosed synthetic and small-sample limits.

The summary covers every expected arm with zero missing cells. All three exported PNG plots were visually inspected: titles, axes, legends, and curves are readable and unclipped; matching PDF exports are present. The generated [results summary](../results/pilot/summary/results_summary.md), [per-task table](../results/pilot/summary/aggregate_by_task.csv), and [paired differences](../results/pilot/summary/paired_deltas.csv) derive only from the corrected prediction file. Their manifest records its hash and the summarizer version.

## Fairness and implementation scope

- Every method starts from the same **full-prefill** cache. The downstream question is absent during selection, including the SnapKV observation window. General task instructions are present in the prefix, so the setting is downstream-question-agnostic, not task-agnostic.
- Cache budgets are exact physical token counts per KV head. Common initial/final reserves count inside the budget. Streaming fills the rest with recent entries; ranked methods use four initial and 32 recent entries before ranking.
- The local SnapKV scorer uses causal attention from 32 tail-context queries, mean GQA aggregation, and width-five average pooling. This matches the inspected KVPress recipe apart from its chosen observation-window size and shared reserves; KVPress defaults to 64 queries. It should remain labeled **SnapKV-style, question hidden**. [KVPress scorer](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/snapkv_press.py).
- The initial-plus-recent baseline retains original RoPE positions and performs one post-prefill compaction. It is not the original StreamingLLM online algorithm with position rerotation. [KVPress's own reproduction note](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/streaming_llm_press.py).
- SVDD, KeyDiff, K-norm, random, and leverage use the same physical gathering and ordinary attention thereafter. Coefficients never become attention weights. Fixed-budget selection can exclude positive SV coefficients and can fill spare budget with zero-coefficient keys.
- When SVDD scores tie at zero, earlier positions are selected first. Thus a large-budget SVDD result can include an implicit early-context fill rule. Selection diagnostics expose zero-score tokens retained; an observed improvement cannot automatically be attributed entirely to support vectors.
- Full-prefill construction plus blockwise scoring is not bounded-memory streaming. Later cached representations can already contain information mixed from earlier tokens. The experiment tests cache compaction, not deletion from all model state or retraining-equivalent forgetting.
- Tensor payload byte savings are real compacted-array sizes. The shared harness keeps the full reference cache and does not provide isolated process-peak memory savings. The cost estimate charges full score-computation time for each independently deployable budget, despite reusing scores between budget trials.

The runtime test suite separately checks stock-cache agreement, per-head gathers, preserved absolute positions, suffix causal masking, context-only SnapKV scoring, and pre-RoPE inversion. This audit inspected those tests and their intended controls; it did not repeat GPU work.

## Claims supported by this design

A completed run can support statements about **this frozen 4-bit-weight Qwen2.5-7B checkpoint, these 54 prompts, this MLX implementation, these selector settings, and 10/20/50% retained prefix caches**. It characterizes the local quality/cost tradeoff under these settings. Paired case differences and full-cache-correct retrieval diagnostics help separate compression damage from existing model failures.

It cannot establish state of the art, universal query-agnostic superiority, exact lossless softmax eviction, a full RULER/LongBench score, independent 54-document statistical evidence, or robustness across modern model families. The paired quality harness alone cannot establish serving speedups or savings in isolated peak process memory. The separately executed fresh-process systems suite supplies local MLX timing and memory measurements under its own reserved-buffer protocol; those do not establish CUDA or general serving performance. The paired bootstrap is exploratory and does not correct shared-filler dependence or multiple comparisons.
