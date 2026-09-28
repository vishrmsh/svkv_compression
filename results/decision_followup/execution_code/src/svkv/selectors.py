"""Query-independent KV selectors and common exact-budget selection.

The ``svdd`` method is a *blockwise SVDD ranker*, not a certified eviction
algorithm. Its coefficients never replace the model's softmax weights.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np


SVDD_BLOCK_SIZE = 256
SVDD_NU = 0.05
SVDD_TOL = 1e-5
SVDD_MAX_ITER = 10_000
GAMMA_SAMPLE_SIZE = 256
LEVERAGE_SKETCH_DIMENSION = 48
LEVERAGE_RIDGE = 0.01
METHODS = ("svdd", "svdd_decision", "keydiff", "leverage", "knorm", "random", "streaming", "snapkv")


def _validate_keys(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys)
    if keys.ndim != 3 or min(keys.shape) < 1:
        raise ValueError("keys must have nonempty shape [heads, tokens, dimensions]")
    if not np.issubdtype(keys.dtype, np.number) or np.iscomplexobj(keys):
        raise ValueError("keys must be real numbers")
    if not np.isfinite(keys).all():
        raise ValueError("keys must contain only finite values")
    return keys


def _normalize(keys: np.ndarray) -> np.ndarray:
    # float64 also avoids an implicit libsvm copy for each fitted block.
    keys = np.asarray(keys, dtype=np.float64)
    return keys / np.maximum(np.linalg.norm(keys, axis=-1, keepdims=True), 1e-12)


def _median_gamma(normalized: np.ndarray, seed: int) -> tuple[float, dict[str, Any]]:
    """Use at most 256 keys, never a context-sized pairwise matrix."""
    n_tokens = normalized.shape[0]
    rng = np.random.default_rng(seed)
    if n_tokens > GAMMA_SAMPLE_SIZE:
        chosen = np.sort(rng.choice(n_tokens, GAMMA_SAMPLE_SIZE, replace=False))
        sample = normalized[chosen]
    else:
        sample = normalized
    squared_norm = np.einsum("nd,nd->n", sample, sample)
    dist2 = np.maximum(
        squared_norm[:, None] + squared_norm[None, :] - 2 * (sample @ sample.T),
        0,
    )
    distances = dist2[np.triu_indices(len(sample), k=1)]
    positive = distances[distances > 1e-12]
    # Identical keys have a constant Gram matrix for every positive gamma.
    fallback = len(positive) == 0
    median = float(np.median(positive)) if not fallback else None
    gamma = 1.0 / median if median is not None else 1.0
    return gamma, {
        "gamma_sample_size": int(len(sample)),
        "median_positive_squared_distance": median,
        "gamma_fallback_identical_keys": fallback,
    }


def _svdd_scores(
    keys: np.ndarray, seed: int, *, decision: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import OneClassSVM

    n_heads, n_tokens, _ = keys.shape
    normalized = _normalize(keys)
    scores = np.zeros((n_heads, n_tokens), dtype=np.float64)
    per_head: list[dict[str, Any]] = []
    failed_blocks = 0
    total_blocks = 0
    total_supports = 0
    largest_block = min(SVDD_BLOCK_SIZE, n_tokens)
    for head in range(n_heads):
        gamma, gamma_info = _median_gamma(normalized[head], seed + head)
        head_fits: list[dict[str, Any]] = []
        head_supports = 0
        for start in range(0, n_tokens, SVDD_BLOCK_SIZE):
            stop = min(start + SVDD_BLOCK_SIZE, n_tokens)
            block_length = stop - start
            estimator = OneClassSVM(
                kernel="rbf",
                gamma=gamma,
                nu=SVDD_NU,
                tol=SVDD_TOL,
                max_iter=SVDD_MAX_ITER,
                cache_size=16,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(normalized[head, start:stop])
            dual = np.asarray(estimator.dual_coef_[0], dtype=np.float64)
            dual_sum = float(dual.sum())
            if dual_sum <= 0 or not np.isfinite(dual_sum):
                raise RuntimeError("OneClassSVM returned invalid dual coefficients")
            # libsvm's one-class dual sum is nu*m. Normalizing gives the
            # SVDD simplex coefficients for the unit-diagonal RBF kernel.
            alpha = dual / dual_sum
            if decision:
                # For the unit-diagonal kernel, d_phi(x)^2 - R^2 equals
                # 2*rho - 2*sum_i(alpha_i*K(x_i,x)). libsvm's decision
                # function uses unnormalized duals and rho; divide by their
                # actual sum, including for a short final block. Each block
                # retains its own fitted boundary. These raw signed scores
                # are ranked globally, with no block quota or rank transform.
                block_scores = -2.0 * estimator.decision_function(
                    normalized[head, start:stop]
                ).reshape(-1) / dual_sum
                scores[head, start:stop] = block_scores
            else:
                scores[head, start + estimator.support_] = block_length * alpha
            convergence_warning = any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
            converged = estimator.fit_status_ == 0 and not convergence_warning
            failed_blocks += int(not converged)
            total_blocks += 1
            head_supports += len(estimator.support_)
            total_supports += len(estimator.support_)
            head_fits.append(
                {
                    "start": int(start),
                    "length": int(block_length),
                    "supports": int(len(estimator.support_)),
                    "converged": bool(converged),
                    "iterations": int(estimator.n_iter_),
                    "raw_dual_sum": dual_sum,
                }
            )
            if decision:
                # A free support vector lies on the fitted boundary. It is
                # not guaranteed to receive a unique decision score.
                free = (dual > 1e-7) & (dual < 1.0 - 1e-7)
                boundary_scores = block_scores[estimator.support_[free]]
                boundary_tolerance = 4.0 * SVDD_TOL / dual_sum
                head_fits[-1].update(
                    {
                        "normalized_rho": float(estimator.offset_[0] / dual_sum),
                        "free_supports": int(free.sum()),
                        "free_support_max_abs_decision_score": (
                            float(np.max(np.abs(boundary_scores)))
                            if boundary_scores.size else None
                        ),
                        "boundary_tie_tolerance": float(boundary_tolerance),
                        "free_supports_within_boundary_tolerance": int(
                            np.count_nonzero(np.abs(boundary_scores) <= boundary_tolerance)
                        ),
                    }
                )
        per_head.append(
            {
                "head": int(head),
                "gamma": float(gamma),
                **gamma_info,
                "raw_support_fraction": float(head_supports / n_tokens),
                "blocks": head_fits,
            }
        )
    return scores, {
        "method_description": (
            "blockwise SVDD signed squared-radius-excess ranker"
            if decision else "blockwise SVDD dual-coefficient ranker"
        ),
        "global_svdd_solve": False,
        "global_ranking_across_blocks": True,
        "exact_softmax_eviction_certificate": False,
        "block_size": SVDD_BLOCK_SIZE,
        "nu": SVDD_NU,
        "tol": SVDD_TOL,
        "max_iter": SVDD_MAX_ITER,
        "normalization": "L2 keys; zero vectors remain zero",
        "gamma_rule": "1 / median positive squared distance, sampled per head",
        "score_rule": (
            "-2 * OneClassSVM.decision_function(X) / sum(raw_dual) = d_phi_squared - R_squared"
            if decision else "block_length * normalized_positive_dual"
        ),
        "solver_converged": failed_blocks == 0,
        "solver_unconverged_blocks": failed_blocks,
        "solver_total_blocks": total_blocks,
        "max_kernel_block_entries": int(largest_block**2),
        "raw_support_fraction": float(total_supports / scores.size),
        "per_head": per_head,
    }


def _leverage_scores(keys: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate ridge leverage on keys already recovered to pre-RoPE space.

    Uses the Gaussian-sketch scoring formula in kvpress LeverageScorePress;
    raw scores and kvpress's global z-scores have identical within-head ranks.
    """
    from scipy.linalg import solve_triangular

    n_heads, n_tokens, dimensions = keys.shape
    rank = min(LEVERAGE_SKETCH_DIMENSION, dimensions)
    rng = np.random.default_rng(seed)
    scores = np.empty((n_heads, n_tokens), dtype=np.float64)
    identity = np.eye(rank)
    for head in range(n_heads):
        head_keys = np.asarray(keys[head], dtype=np.float64)
        centered = head_keys - head_keys.mean(axis=0, keepdims=True)
        projection = rng.normal(size=(dimensions, rank)) / np.sqrt(rank)
        sketch = centered @ projection
        gram = sketch.T @ sketch
        # Symmetrization also matches the reference's treatment of roundoff.
        gram = (gram + gram.T) * 0.5 + LEVERAGE_RIDGE * identity
        cholesky = np.linalg.cholesky(gram)
        whitened = solve_triangular(cholesky, sketch.T, lower=True, check_finite=False)
        scores[head] = np.einsum("rn,rn->n", whitened, whitened)
    return scores, {
        "method_description": "Gaussian-sketched ridge leverage on centered pre-RoPE keys",
        "key_representation_expected": "pre_rope; caller removes RoPE before scoring",
        "key_l2_normalization": False,
        "sequence_centered": True,
        "sketch_dimension_requested": LEVERAGE_SKETCH_DIMENSION,
        "sketch_dimension": int(rank),
        "ridge": LEVERAGE_RIDGE,
        "projection_distribution": "independent Gaussian entries N(0, 1 / sketch_dimension)",
        "score_rule": "squared column norm of solve(cholesky(X.T@X + ridge*I), X.T)",
        "score_normalization": "none; global z-normalization preserves within-head ordering",
        "max_gram_entries": int(rank**2),
        "arithmetic": "float64",
    }


def compute_scores(
    keys: np.ndarray, method: str, seed: int = 0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return [H, N] scores, with larger scores preferred for retention.

    Compute once per layer and reuse across cache budgets. ``snapkv`` needs
    scores from the model's observation-window attention; supply those directly
    to :func:`select_indices` instead. ``leverage`` expects already-pre-RoPE
    keys; the caller is responsible for inversion. No selector here receives
    model queries.
    """
    started = time.perf_counter()
    keys = _validate_keys(keys)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    n_heads, n_tokens, _ = keys.shape
    diagnostics: dict[str, Any] = {"method": method, "seed": int(seed)}
    if method in ("svdd", "svdd_decision"):
        scores, details = _svdd_scores(keys, seed, decision=method == "svdd_decision")
        diagnostics.update(details)
    elif method == "leverage":
        scores, details = _leverage_scores(keys, seed)
        diagnostics.update(details)
    elif method == "keydiff":
        normalized = _normalize(keys)
        anchor = normalized.mean(axis=1, keepdims=True)
        raw_keys = np.asarray(keys, dtype=np.float64)
        # Match torch F.cosine_similarity's 1e-8 norm floor, separately from
        # F.normalize's 1e-12 floor used to construct the mean anchor.
        cosine_keys = raw_keys / np.maximum(np.linalg.norm(raw_keys, axis=-1, keepdims=True), 1e-8)
        cosine_anchor = anchor / np.maximum(np.linalg.norm(anchor, axis=-1, keepdims=True), 1e-8)
        scores = -np.einsum("hnd,hqd->hn", cosine_keys, cosine_anchor)
        diagnostics["method_description"] = "negative cosine to mean normalized key"
    elif method == "knorm":
        norm = np.linalg.norm(np.asarray(keys, dtype=np.float64), axis=-1)
        scores = 1.0 / np.maximum(norm, 1e-12)
        diagnostics["method_description"] = "inverse raw key L2 norm"
    elif method == "random":
        scores = np.random.default_rng(seed).random((n_heads, n_tokens))
        diagnostics["method_description"] = "seeded uniform random scores"
    elif method == "streaming":
        scores = np.broadcast_to(np.arange(n_tokens, dtype=np.float64), (n_heads, n_tokens)).copy()
        diagnostics["method_description"] = "initial sinks plus most recent remaining tokens"
    else:
        raise ValueError("snapkv requires externally computed observation-window scores")
    diagnostics["score_seconds"] = time.perf_counter() - started
    return scores, diagnostics


def _reserved_indices(n_tokens: int, budget: int, sinks: int, recent: int) -> np.ndarray:
    # For tiny budgets reserve sinks first, then the newest positions that fit.
    sink_count = min(sinks, budget, n_tokens)
    recent_count = min(recent, budget - sink_count, n_tokens - sink_count)
    return np.concatenate(
        (np.arange(sink_count), np.arange(n_tokens - recent_count, n_tokens))
    ).astype(np.int64)


def select_indices(
    keys: np.ndarray,
    budget: int,
    method: str,
    scores: np.ndarray | None = None,
    seed: int = 0,
    recent: int = 32,
    sinks: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select exactly ``budget`` distinct chronological positions per KV head.

    Ranked methods reserve initial ``sinks`` and final ``recent`` positions.
    Streaming reserves sinks and fills its entire remaining budget with recent
    positions. Equal scores prefer earlier positions. These are original cache
    positions: the caller must preserve the model's original RoPE positions.
    """
    started = time.perf_counter()
    keys = _validate_keys(keys)
    n_heads, n_tokens, _ = keys.shape
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise ValueError("budget must be an integer")
    if not 0 <= budget <= n_tokens:
        raise ValueError("budget must lie between zero and the number of tokens")
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0 for x in (sinks, recent)):
        raise ValueError("sinks and recent must be nonnegative integers")
    diagnostics: dict[str, Any] = {"method": method, "seed": int(seed)}
    if method == "streaming":
        reserved = _reserved_indices(n_tokens, budget, sinks, budget)
        indices = np.broadcast_to(reserved, (n_heads, budget)).copy()
        diagnostics["scores_source"] = "not_used"
    else:
        if scores is None:
            scores, score_diagnostics = compute_scores(keys, method, seed=seed)
            diagnostics.update(score_diagnostics)
            diagnostics["scores_source"] = "computed"
        else:
            diagnostics["scores_source"] = "provided"
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (n_heads, n_tokens) or not np.isfinite(scores).all():
            raise ValueError("scores must be finite and have shape [heads, tokens]")
        if method == "svdd" and np.any(scores < 0):
            raise ValueError("svdd scores must be nonnegative dual-coefficient scores")
        reserved = _reserved_indices(n_tokens, budget, sinks, recent)
        available = np.ones(n_tokens, dtype=bool)
        available[reserved] = False
        candidates = np.flatnonzero(available)
        n_ranked = budget - len(reserved)
        indices = np.empty((n_heads, budget), dtype=np.int64)
        for head in range(n_heads):
            # Stable sorting over chronologically ordered candidates makes ties
            # reproducible and yields nested sets as budgets grow.
            order = np.argsort(-scores[head, candidates], kind="stable")
            indices[head] = np.sort(np.concatenate((reserved, candidates[order[:n_ranked]])))
        if method == "svdd":
            support_counts = np.count_nonzero(scores > 0, axis=1)
            kept_counts = np.count_nonzero(np.take_along_axis(scores, indices, axis=1) > 0, axis=1)
            discarded = support_counts - kept_counts
            diagnostics.update(
                {
                    "global_svdd_solve": False,
                    "exact_softmax_eviction_certificate": False,
                    "raw_support_fraction": float(support_counts.sum() / scores.size),
                    "raw_support_fraction_per_head": (support_counts / n_tokens).tolist(),
                    "supports_discarded": int(discarded.sum()),
                    "supports_discarded_per_head": discarded.tolist(),
                    "supports_retained": int(kept_counts.sum()),
                    "zero_score_tokens_retained": int(n_heads * budget - kept_counts.sum()),
                    "retains_all_block_supports": bool(np.all(discarded == 0)),
                }
            )
        elif method == "svdd_decision":
            diagnostics.update(
                {
                    "global_svdd_solve": False,
                    "global_ranking_across_blocks": True,
                    "exact_softmax_eviction_certificate": False,
                    "signed_decision_scores": True,
                    "score_sign_identifies_support": False,
                }
            )
    diagnostics.update(
        {
            "tokens_before": int(n_tokens),
            "tokens_after_per_head": int(budget),
            "retained_fraction": float(budget / n_tokens),
            "reserved_per_head": int(len(reserved)),
            "sinks_requested": int(sinks),
            "recent_requested": int(recent),
            "selection_seconds_including_score_if_computed": time.perf_counter() - started,
        }
    )
    return indices, diagnostics
