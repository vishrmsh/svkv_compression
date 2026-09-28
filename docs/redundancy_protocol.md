# Redundancy comparison protocol

This comparison re-evaluates the original SV Attention selection probes against
KeyDiff and exact linear and kernel ridge leverage. It tests whether the reported
advantage over similarity proxies persists against geometric selectors. The
original examples and published results are already known, so this is not a
held-out benchmark.

The parameter manifest is [redundancy_check.json](../configs/redundancy_check.json).
All three new methods are included unconditionally, before inspecting results:
KeyDiff scoring, exact linear ridge leverage, and exact RBF kernel ridge leverage.
There is no bandwidth, ridge, budget, or seed search.

## Original synthetic

The [SV Attention implementation](https://github.com/VyLabs-AI/sv-attention)
defines 60 trials from one
`numpy.random.RandomState(0)` stream. Each context contains 66 four-dimensional
keys: six singleton groups and six groups with ten near-duplicates each. Twelve
Gaussian centers have scale 1.5; keys have Gaussian jitter 0.15. There are two
queries per center with jitter 0.05. The second query of each pair is evaluated;
the first is visible only to the warm-query similarity-mass comparator. The
oracle mass comparator sees both. These are custom similarity-sum proxies,
not implementations of the published H2O system.

The RBF is `exp(-squared_distance / 1.2**2)`. SVDD uses `nu=0.45`, a full 66×66
Gram, CLARABEL absolute/relative gap and feasibility tolerances `1e-11`, and
partition threshold `1e-6`. The original bordered solve is reproduced after
partitioning; an empty margin set is a recorded failure. Every non-full method
keeps that trial's complete SVDD support count. No reweighting is applied: each
method uses identical uniform RBF attention over retained keys into group
one-hot values. The primary metric is mean singleton-group recall; all-group
recall and group coverage are secondary diagnostics.

The existing random baseline consumes the same draws in the same order as the
original script. Added methods consume none of this stream. On a solver
failure, the original skipped query/random draws are likewise skipped.
Failed attempts remain in the attempt/status counts; metric means and paired
intervals use completed trials, so failures are not counted as zero recall. Original
proxy sorting remains NumPy quicksort; new methods use a prespecified stable
position tie break. The original order puts singleton keys first, so ties
could favor rare groups; exact ties and geometry remain limitations of this
synthetic. Published rounded targets are SVDD 0.861, oracle mass 0.319, random
0.472, and recency 0.000. Agreement is checked, never forced by changing settings.

## Original held-out-SpO2 ICU probe

The credentialed `icu_vitals_n1500.npz` cache is read without alteration.
The first 1,465 sequences are attempted in their original order. Events are
the cached, unstandardized SpO2 values below 90 after the original within-stay
forward/backward filling and cohort-median imputation. SpO2 is removed before
any selector receives inputs, leaving five standardized vital channels.

SVDD uses `nu=0.3`, RBF width 3, CLARABEL gap/feasibility/infeasibility
tolerances `1e-10`, and partition threshold `1e-7`, matching the original
`FastOneClassSVM.seed_from_qp`. Its later maintenance inverse and gradients are
irrelevant to the support indices and are not constructed here. Original
eligibility requires an event-positive stay, successful solve, and support
count strictly between zero and the number of tokens. All statuses are
reported, including no-event stays, numerical failures, and boundary support
counts. Original comparison targets are 718 completed stays, mean retention
budget 0.32260248, SVDD event retention 0.46437441, and density-proxy retention
0.22537540. The density score sums off-diagonal RBF similarities. Random
selection uses `RandomState(number_of_tokens)` separately for each stay;
recency retains the last budgeted hours.

Primary retention is the unweighted mean across eligible stays of the
fraction of event hours retained. Secondary retrieval asks whether each event
hour's nearest retained key is also labeled an event, under the original RBF
argmax rule and chronological ties. Even full-context retrieval can be below
one when different event labels share the same remaining five-channel key.
These are record-selection outcomes, not clinical-prediction performance.

## Added selectors and inference

- **KeyDiff:** negative cosine between a key and the mean of L2-normalized keys,
  matching the score used by the KV pilot. This is a scoring adaptation, not
  KeyDiff's complete cache-management procedure.
- **Linear ridge leverage:** for centered original keys `Z`, compute
  `diag(Z @ inverse(Z.T @ Z + 0.01 I) @ Z.T)`. Exact four- or five-dimensional
  Cholesky solves avoid random sketches. Keys are not L2-normalized for this
  method; the original synthetic/standardized clinical coordinates are used.
- **Kernel ridge leverage:** `diag(K @ inverse(K + 0.01 I))`, where `K` is the
  exact, uncentered RBF Gram used by SVDD. This controls for nonlinear kernel
  geometry when assessing any residual SVDD advantage. Ridge 0.01 is absolute,
  added to the unnormalized Gram, with no division by the sample count.

All selectors are query-free except the explicitly labeled original mass
comparators. No new selector sees group/event labels. Each score is ranked
globally within the tiny context or stay, with no sinks, recency reserves,
blocks, truncation of SVDD supports, or zero-score fills. The full method is
an uncompressed reference and is explicitly exempt from matched budgets. It
is not an accuracy ceiling: retaining all dense duplicates can bias the
uniform RBF readout away from singleton groups. The frozen run manifest's
phrase "full-context ceiling" should be read as "full-context reference";
it does not describe an upper bound on recall.

Mean differences use paired 10,000-resample percentile intervals, seed 0,
with contexts or stays as resampling units. All method contrasts are reported
without multiplicity correction; they are descriptive, not confirmatory
significance tests. Stays are assumed independent as in the original analysis;
multiple stays can belong to one patient, and patient IDs are not loaded for
clustering. A win against linear leverage alone would not identify an
SVDD-specific mechanism; kernel leverage is included from the outset. Even a
win against both on these reused protocols would support only a scoped finding.

## Reproduction, provenance, and privacy

```sh
python -m venv .venv-redundancy
.venv-redundancy/bin/pip install -r requirements-redundancy-lock.txt
.venv-redundancy/bin/pip install --no-deps -e .
.venv-redundancy/bin/python scripts/run_redundancy_check.py \
  --clinical-cache /path/to/credentialed/icu_vitals_n1500.npz \
  --output results/redundancy_reproduction
```

Run the synthetic alone with `--task synthetic`; it needs no clinical files.
Use a new output directory for any repeat. The runner archives the executed
source/configuration, package versions, source hashes, and input-cache hash.
It does not use wall time as an isolated comparison of selector cost.

The standalone module independently implements the protocols and mathematical
formulas in [VyLabs-AI/sv-attention](https://github.com/VyLabs-AI/sv-attention),
whose release uses Apache-2.0. The execution manifest records source provenance
and SHA-256 hashes. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

The cached NPZ uses object arrays and therefore requires `allow_pickle=True`;
only a trusted credentialed local cache should be supplied. Stay identifiers
are never accessed. Clinical features, labels, per-stay metrics, selected
indices, and solver arrays remain in memory; only cohort-level aggregate
results are written. Synthetic trial rows and selected indices are safe to
publish. Clinical caches and patient-level working files must not be uploaded.
