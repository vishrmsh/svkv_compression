# Pilot protocol and interpretation

This is a small-scale evaluation of KV-cache selectors. The main protocol is
recorded in [`configs/pilot_protocol.json`](../configs/pilot_protocol.json).
The exploratory 8k smoke case in `results/smoke` is excluded from the main
aggregate. After that implementation check, the literature review motivated
adding pre-RoPE SVDD and leverage scoring before the main run. No solver
hyperparameters were selected using benchmark answers.

An independent audit during an initial partial execution found a missing newline
between the synthetic prose and question. That execution was stopped and
preserved under `results/pre_separator_check`, excluded from every final table.
An explicit separator was added to the question suffix, and every arm was rerun
from scratch. Prefixes, sample selection, solver settings and scoring rules did
not change.

## Model and data

- Qwen2.5-7B-Instruct, MLX community four-bit **weight** conversion, pinned
  revision `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`. The KV cache remains
  16-bit. This is a widely used older model, not a claim about current frontier
  model quality. The tested model uses 28 layers, four KV heads, 28 query heads,
  head dimension 128, full attention and unscaled RoPE.
- Thirty-six custom natural-prose NIAH cases: single and two-number retrieval,
  nominal 8,192/16,384/32,768-token windows, seeds 17/29 and depths .1/.5/.9.
  Prefixes reserve 256 tokens for the question and generation. The distractor
  prose comes from the first three Qasper contexts (11,435 tokens), cyclically
  repeated and rotated to construct longer contexts. These are **not official
  RULER samples**. Their conspicuous magic-number wording can favor outliers.
- Eighteen real LongBench cases: six each from HotpotQA, Qasper and English
  passage retrieval. Deterministically shuffle with seed 20260927 and accept
  the first six untruncated contexts of 7,900–32,000 prefix tokens per task.
  Questions and generation must still fit the 32,768-token model limit.
  Sampling uses length, never full-cache correctness or answer content.
- The LongBench archive is pinned to revision
  `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`. Prompts use a common Qwen chat
  wrapper and short task instructions; therefore these are subset results
  with adapted prompts, not official leaderboard scores.

The local `data/pilot_cases.jsonl` contains the exact token IDs. It is excluded
from Git along with the public source archive; `data/case_manifest.json`
records source row IDs, generation parameters and token hashes. Run
`scripts/prepare_data.py` to reconstruct the file from the pinned assets.

## Paired comparison

Each context is prefetched once with full native attention. All methods start
from the identical resulting cache, and compress **before the downstream
question enters the model**. The context states may already contain propagated
information; this tests practical post-prefill compression, not removal from
all computation that created those states.

Every ranked method reserves four initial tokens and the last 32 context
tokens inside its total budget. Each layer and KV head retains exactly
`floor(prefix_length * fraction)` original positions for fractions .1/.2/.5.
Original rotated K/V tensors are gathered into smaller physical arrays, then
the question and generated tokens are appended without further eviction.
The effective fraction grows slightly during decoding; before- and after-
generation byte counts are both logged. The full-cache arm is run on every
case. There are 25 arms per context, 1,350 answers in total.

The selectors are defined in [`selector_method.md`](selector_method.md).
`svdd_prerope` and `leverage` invert the native key rotation for scoring only;
this recovers pre-RoPE geometry to the precision allowed by earlier rounding.
The serving cache always keeps its original rotated keys. SnapKV-style scores
use the final 32 **context** queries, causal attention, GQA group averaging and
a width-five average pool, following the KVPress recipe. This is a restricted
question-hidden adaptation, not evidence against question-aware SnapKV. The
`streaming` arm uses sinks plus recent tokens with the same absolute positions
as all arms; it does not reproduce StreamingLLM's position remapping or online
eviction schedule. KeyDiff and leverage are independently implemented scorer
adaptations, not full reproductions of their original systems.

SVDD sees keys only. Its 256-token block solves are separate local objectives;
the sampled per-head bandwidth uses the entire already available context.
Consequently it is query-independent but is not a streaming, bounded-prefix-
memory algorithm. All arms retain the full prefill computation cost.

## Measurements and scoring

Generation is greedy, with 32-token maximum for custom NIAH, 16 for passage
retrieval and 128 for natural QA. NIAH score is the fraction of requested
seven-digit answers found with numeric boundaries; order and extra guesses are
not penalized. A two-needle score of .5 means one of two numbers was recalled,
not that the whole answer was exactly correct. Natural QA uses the maximum
normalized token F1 over accepted gold answers. Passage retrieval uses the
LongBench fraction of predicted paragraph numbers equal to the gold number.
These different metrics are reported by task. Any across-task mean is only an
exploratory equal-example composite, not a canonical benchmark score.

All raw answers, gold answers, token counts, exact KV bytes and per-case
timings are saved in JSONL. Solver files expose support fractions and
convergence, including failed solves if any. Timing separates shared prefill,
key export, score construction, ranking/gather and answer generation. Scoring
is reused across three budgets for efficiency; the estimated cost of deploying
one method charges its whole score construction once. This estimate is not
an independent end-to-end timing experiment. In particular the shared prefill
includes a small tail-query capture used by SnapKV; other methods do not need
that capture in deployment.

The paired harness intentionally holds the original full cache while testing
each arm. Its allocator peak and RSS are diagnostic values, **not measurements
of deployment memory savings**. Compact KV tensor bytes are actual array sizes,
but excluding weights, selector state, allocator caches and temporary buffers.
The isolated systems script releases the full-cache reference before measuring
decode storage and uses a fixed number of forward steps. It also uses a
separately validated reserved append-buffer cache for every arm, including full
cache. The quality harness uses concatenating arrays; its timing can include
full-array copy costs and is not the source of the final decode speed comparison.
The isolated backend allocates space for the known prefix, question and 32
decode steps, then writes appends into that capacity. Both logical KV payload
and actually allocated capacity, including unused reserve, are reported. Short
real-model checks found exactly matching logits for full and compacted caches
between the two backends. Full-context prefill still limits peak-memory savings
in this prototype.

Report paired differences with fixed-seed bootstrap intervals as exploratory
uncertainty, with explicit sample sizes. Cases share templates/distractor
sources, so independence and generalization are limited. Also report retrieval
on the common subgroup where the full-cache arm has score one; do not discard
base-model failures from the main table.

## Correctness and scope

The tests check exact budgets, solver behavior and a counterexample to
softmax losslessness. Opt-in real-model Metal checks cover stock/full-budget
logit equality, head-specific gathers, physical versus absolute positions,
causal masking, and SnapKV scores against an independent float64 reference.
Quantized batched and single-token kernels need not be bitwise identical;
the tests compare distributions and exact masks separately.

The results characterize the tested implementation and settings. They do not
establish significance across model families, superiority over official CUDA
systems, streaming stability, or novelty. H2O, PyramidKV, TOVA, full Compactor,
KV quantization, official RULER, and additional model families are outside
the scope of this evaluation.

The earlier SV deletion theorem does not apply to ordinary softmax. No result
here should be described as certified lossless KV eviction.
