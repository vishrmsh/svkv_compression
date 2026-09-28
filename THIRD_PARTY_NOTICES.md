# Sources and third-party assets

This standalone pilot does not import or modify `plain_jane`. Its algorithms
were written for this workspace. The older project was read as a reference;
the audit in `docs/prior_work_status.md` records the lineage and limitations.

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

See `references.bib` and `docs/literature_review.md` for paper citations. No
model weights, private clinical data, credentials, or copied private project
source are intended for publication in this repository.
