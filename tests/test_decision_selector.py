"""Independent kernel reference and global-budget checks for the fixed ablation."""

import numpy as np
import pytest
from sklearn.svm import OneClassSVM

from svkv.selectors import compute_scores, select_indices


def _reference(keys, fit_diagnostics):
    """Reconstruct distances and fitted radius without decision_function."""
    normalized = keys / np.maximum(np.linalg.norm(keys, axis=-1, keepdims=True), 1e-12)
    reference = np.empty(keys.shape[:2])
    alpha_reference = np.zeros(keys.shape[:2])
    supports = 0
    free_residuals = []
    for head_details in fit_diagnostics["per_head"]:
        head = head_details["head"]
        gamma = head_details["gamma"]
        for block_details in head_details["blocks"]:
            start = block_details["start"]
            length = block_details["length"]
            block = normalized[head, start:start + length]
            fit = OneClassSVM(
                kernel="rbf", gamma=gamma, nu=0.05, tol=1e-5,
                max_iter=10_000, cache_size=16,
            ).fit(block)
            alpha = np.zeros(length)
            raw_dual = fit.dual_coef_[0]
            dual_sum = raw_dual.sum()
            alpha[fit.support_] = raw_dual / dual_sum
            dist2 = ((block[:, None] - block[None, :]) ** 2).sum(axis=-1)
            kernel = np.exp(-gamma * dist2)
            center_squared_norm = alpha @ kernel @ alpha
            squared_distance_to_center = 1.0 - 2.0 * (kernel @ alpha) + center_squared_norm
            normalized_rho = -fit.intercept_[0] / dual_sum
            radius_squared = 1.0 - 2.0 * normalized_rho + center_squared_norm
            reference[head, start:start + length] = squared_distance_to_center - radius_squared
            alpha_reference[head, start:start + length] = length * alpha
            supports += len(fit.support_)
            free = (alpha > 1e-7) & (alpha < 1 / (0.05 * length) - 1e-7)
            free_residuals.append(reference[head, start:start + length][free])
    return reference, alpha_reference, supports, free_residuals


def test_decision_score_matches_explicit_svdd_radius_including_short_block():
    keys = np.random.default_rng(716).normal(size=(2, 301, 8))
    scores, fit = compute_scores(keys, "svdd_decision", seed=23)
    reference, alpha_reference, support_count, _ = _reference(keys, fit)
    np.testing.assert_allclose(scores, reference, rtol=1e-8, atol=2e-15)
    assert [block["length"] for block in fit["per_head"][0]["blocks"]] == [256, 45]
    assert fit["raw_support_fraction"] == pytest.approx(support_count / scores.size)
    assert fit["raw_support_fraction"] != pytest.approx(np.mean(scores > 0))
    assert fit["solver_total_blocks"] == 4
    assert fit["solver_converged"]
    assert fit["global_ranking_across_blocks"]
    assert not fit["global_svdd_solve"]
    assert not fit["exact_softmax_eviction_certificate"]
    assert scores.min() < 0 < scores.max()

    # Original alpha ranking retains exactly the previous normalized-dual
    # definition. The added decision scorer does not alter its fitting path.
    alpha_scores, alpha_fit = compute_scores(keys, "svdd", seed=23)
    np.testing.assert_array_equal(alpha_scores, alpha_reference)
    np.testing.assert_allclose(alpha_scores[:, :256].sum(axis=1), 256)
    np.testing.assert_allclose(alpha_scores[:, 256:].sum(axis=1), 45)
    assert alpha_fit["raw_support_fraction"] == fit["raw_support_fraction"]


def test_global_signed_ranking_has_no_equal_block_quota():
    keys = np.ones((1, 512, 3))
    # Every score in the second block beats every score in the first block,
    # although all scores are negative. A global 64-token budget retains
    # all 64 from that block. Score sign is unrelated to raw support status.
    scores = np.concatenate((np.full(256, -3.0), np.full(256, -1.0)))[None]
    indices, diagnostics = select_indices(
        keys, 64, "svdd_decision", scores=scores, sinks=0, recent=0,
    )
    np.testing.assert_array_equal(indices, np.arange(256, 320)[None])
    assert diagnostics["global_ranking_across_blocks"]
    assert not diagnostics["score_sign_identifies_support"]
    assert "raw_support_fraction" not in diagnostics
    assert "supports_discarded" not in diagnostics
    assert "zero_score_tokens_retained" not in diagnostics


def test_free_support_decision_scores_remain_boundary_ties_within_solver_tolerance():
    keys = np.random.default_rng(90).normal(size=(1, 80, 5))
    scores, diagnostics = compute_scores(keys, "svdd_decision", seed=11)
    _, _, _, residuals = _reference(keys, diagnostics)
    block = diagnostics["per_head"][0]["blocks"][0]
    assert block["free_supports"] > 2
    assert residuals[0].size == block["free_supports"]
    assert np.max(np.abs(residuals[0])) <= block["boundary_tie_tolerance"]
    assert block["free_supports_within_boundary_tolerance"] == block["free_supports"]
    assert block["free_support_max_abs_decision_score"] == pytest.approx(
        np.max(np.abs(residuals[0])), abs=2e-15,
    )
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("budget", [0, 3, 36, 60, 301])
def test_decision_exact_budget_and_protected_positions(budget):
    keys = np.random.default_rng(107).normal(size=(2, 301, 8))
    scores, fit = compute_scores(keys, "svdd_decision", seed=17)
    from_scores, supplied = select_indices(keys, budget, "svdd_decision", scores=scores, seed=17)
    direct, diagnostics = select_indices(keys, budget, "svdd_decision", seed=17)
    np.testing.assert_array_equal(from_scores, direct)
    assert direct.shape == (2, budget)
    assert np.all(np.diff(direct, axis=1) > 0)
    if budget:
        np.testing.assert_array_equal(direct[:, :min(4, budget)], np.tile(np.arange(min(4, budget)), (2, 1)))
    if budget >= 36:
        np.testing.assert_array_equal(direct[:, -32:], np.tile(np.arange(269, 301), (2, 1)))
    assert diagnostics["raw_support_fraction"] == fit["raw_support_fraction"]
    assert "raw_support_fraction" not in supplied


def test_identical_zero_keys_and_stable_ties():
    keys = np.zeros((1, 301, 4))
    scores, diagnostics = compute_scores(keys, "svdd_decision")
    np.testing.assert_allclose(scores, 0, atol=1e-14)
    assert diagnostics["solver_converged"]
    # Supplying exact tied values isolates the common sort rule from solver
    # roundoff, which remains deliberately unsnapped in actual scores.
    indices, _ = select_indices(keys, 50, "svdd_decision", scores=np.zeros((1, 301)))
    np.testing.assert_array_equal(indices[0], np.r_[np.arange(18), np.arange(269, 301)])

