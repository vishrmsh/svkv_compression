# Sources and third-party assets

This project implements geometric cache-selection methods and evaluates them
using the dependencies and datasets below. The relationship to earlier
support-vector memory methods is documented in
[docs/prior_work_status.md](docs/prior_work_status.md).

- Inference: [MLX](https://github.com/ml-explore/mlx) and
  [MLX LM](https://github.com/ml-explore/mlx-lm), installed as dependencies.
- Model: [Qwen2.5-7B-Instruct MLX conversion](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit),
  Apache-2.0 according to its model card. Weights are downloaded separately
  and excluded from Git.
- Data: [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench).
  The original archive and generated tokenized samples are excluded from Git;
  downstream users should observe the source dataset terms.
- Scorer formulas: independently implemented from primary papers and the
  [NVIDIA KVPress](https://github.com/NVIDIA/kvpress) reference recipes. The
  implementations here are adaptations; names do not claim full replication.
- Optimization: scikit-learn's LIBSVM-backed OneClassSVM. RBF unit-diagonal
  equivalence to SVDD is explained in `docs/selector_method.md`.
- The redundancy comparison independently reconstructs the original
  [SV Attention](https://github.com/VyLabs-AI/sv-attention) synthetic and
  held-out-vital ICU protocols, distributed by that project under Apache-2.0.
  Source provenance and exact solver settings are recorded in
  `docs/redundancy_protocol.md`. Its small reference QPs use CVXPY and CLARABEL.
  Credentialed ICU inputs remain local; only aggregate ICU results are intended
  for this repository.

See `references.bib` and `docs/literature_review.md` for paper citations. No
model weights, private clinical data, credentials, or copied private project
source are intended for publication in this repository.
