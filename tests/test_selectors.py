"""Selection invariants, geometry, and the boundary of the deletion claim."""

import numpy as np
import pytest

from svkv.selectors import compute_scores, select_indices


@pytest.fixture
def keys():
    return np.random.default_rng(100).normal(size=(2, 301, 8)).astype(np.float32)


@pytest.mark.parametrize("method", ["svdd", "keydiff", "leverage", "knorm", "random", "streaming", "snapkv"])
@pytest.mark.parametrize("budget", [0, 3, 36, 60, 301])
def test_exact_budget_sorted_unique_reserved(keys, method, budget):
    supplied = np.random.default_rng(0).random(keys.shape[:2]) if method == "snapkv" else None
    indices, diagnostics = select_indices(keys, budget, method, scores=supplied)
    assert indices.shape == (2, budget)
    assert indices.dtype == np.int64
    if budget:
        assert np.all((indices >= 0) & (indices < keys.shape[1]))
        assert np.all(np.diff(indices, axis=1) > 0)
        assert np.all(indices[:, :min(4, budget)] == np.arange(min(4, budget)))
    if budget >= 36:
        assert np.all(indices[:, -32:] == np.arange(269, 301))
    assert diagnostics["tokens_after_per_head"] == budget


@pytest.mark.parametrize("method", ["svdd", "keydiff", "leverage", "knorm", "random", "streaming"])
def test_precomputed_equals_direct_and_deterministic(keys, method):
    scores, _ = compute_scores(keys, method, seed=17)
    first, _ = select_indices(keys, 60, method, scores=scores, seed=17)
    second, _ = select_indices(keys, 60, method, seed=17)
    np.testing.assert_array_equal(first, second)


def test_streaming_exact_positions(keys):
    indices, _ = select_indices(keys, 10, "streaming")
    np.testing.assert_array_equal(indices[0], [0, 1, 2, 3, 295, 296, 297, 298, 299, 300])


def test_keydiff_prefers_angular_outlier():
    keys = np.array([[[1, 0], [2, 0], [3, 0], [-1, 0]]], dtype=np.float32)
    indices, _ = select_indices(keys, 1, "keydiff", sinks=0, recent=0)
    np.testing.assert_array_equal(indices, [[3]])


def test_knorm_prefers_small_norm_and_handles_zero():
    keys = np.array([[[3, 0], [0, 0], [1, 0]]], dtype=np.float32)
    indices, _ = select_indices(keys, 2, "knorm", sinks=0, recent=0)
    np.testing.assert_array_equal(indices, [[1, 2]])


def test_leverage_matches_explicit_ridge_hat_matrix():
    keys = np.random.default_rng(45).normal(size=(1, 17, 5))
    seed = 11
    scores, diagnostics = compute_scores(keys, "leverage", seed=seed)
    centered = keys[0] - keys[0].mean(axis=0)
    projection = np.random.default_rng(seed).normal(size=(5, 5)) / np.sqrt(5)
    sketch = centered @ projection
    ridge = diagnostics["ridge"]
    # A small dense hat matrix is an independent reference, not used in the
    # selector. This detects Cholesky transpose / inverse-squaring mistakes.
    hat = sketch @ np.linalg.solve(sketch.T @ sketch + ridge * np.eye(5), sketch.T)
    np.testing.assert_allclose(scores[0], np.diag(hat), rtol=1e-11, atol=1e-12)
    assert np.all((scores >= 0) & (scores <= 1))
    assert scores.sum() <= 5


def test_leverage_is_translation_invariant_and_permutation_equivariant():
    keys = np.random.default_rng(71).normal(size=(2, 59, 12))
    original, _ = compute_scores(keys, "leverage", seed=14)
    offsets = np.random.default_rng(3).normal(size=(2, 1, 12)) * 100
    shifted, _ = compute_scores(keys + offsets, "leverage", seed=14)
    order = np.random.default_rng(11).permutation(59)
    permuted, _ = compute_scores(keys[:, order], "leverage", seed=14)
    np.testing.assert_allclose(shifted, original, atol=1e-10)
    np.testing.assert_allclose(permuted, original[:, order], atol=1e-10)


def test_leverage_preserves_rare_direction_and_has_bounded_rank():
    keys = np.zeros((1, 100, 64))
    keys[0, -1, -1] = 1
    scores, diagnostics = compute_scores(keys, "leverage")
    selected, _ = select_indices(keys, 1, "leverage", scores=scores, sinks=0, recent=0)
    np.testing.assert_array_equal(selected, [[99]])
    assert diagnostics["sketch_dimension"] == 48
    assert diagnostics["max_gram_entries"] == 48**2
    assert diagnostics["ridge"] == 0.01
    assert diagnostics["score_seconds"] >= 0
    constant, _ = compute_scores(np.ones((1, 8, 64)), "leverage")
    np.testing.assert_array_equal(constant, 0)


def test_svdd_retains_rare_direction_and_reports_truncation():
    keys = np.array([[[1, 0]] * 60 + [[-1, 0]] * 4], dtype=np.float32)
    scores, fit = compute_scores(keys, "svdd")
    assert scores[0, 60:].max() > 0
    indices, selected = select_indices(keys, 2, "svdd", scores=scores, sinks=0, recent=0)
    assert np.any(indices >= 60)
    assert fit["solver_converged"]
    assert selected["supports_discarded"] > 0
    assert not selected["exact_softmax_eviction_certificate"]


def test_svdd_chunk_scaling_and_kernel_bound(keys):
    scores, fit = compute_scores(keys, "svdd")
    np.testing.assert_allclose(scores[:, :256].sum(axis=1), 256)
    np.testing.assert_allclose(scores[:, 256:].sum(axis=1), 45)
    assert fit["max_kernel_block_entries"] == 256**2
    assert fit["solver_total_blocks"] == 4
    assert all(len(head["blocks"]) == 2 for head in fit["per_head"])


@pytest.mark.parametrize("nu", [0.05, 0.4])
def test_svdd_matches_independent_constrained_qp_and_kkt(monkeypatch, nu):
    from scipy.optimize import minimize
    import svkv.selectors as selectors

    # The default nu tests the deployed scorer; 0.4 deliberately activates
    # dual box constraints and catches an incorrect one-class/SVDD rescaling.
    monkeypatch.setattr(selectors, "SVDD_NU", nu)
    n = 40
    keys = (np.random.default_rng(871).normal(size=(1, n, 5))
            * np.array([1.0, 0.6, 0.3, 0.2, 0.1])
            + np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    scores, diagnostics = compute_scores(keys, "svdd", seed=25)
    alpha = scores[0] / n
    normalized = keys[0] / np.linalg.norm(keys[0], axis=1, keepdims=True)
    gamma = diagnostics["per_head"][0]["gamma"]
    distances = ((normalized[:, None] - normalized[None, :])**2).sum(axis=-1)
    kernel = np.exp(-gamma * distances)
    cap = 1 / (nu * n)
    # For RBF diagonal=1, minimizing alpha.T K alpha is exactly the SVDD
    # dual after dropping its constant linear term. This optimizer is
    # independent of libsvm and uses the normalized simplex constraints.
    reference = minimize(
        lambda a: a @ kernel @ a,
        np.full(n, 1 / n),
        jac=lambda a: 2 * kernel @ a,
        bounds=[(0, cap)] * n,
        constraints={"type": "eq", "fun": lambda a: a.sum() - 1,
                     "jac": lambda a: np.ones(n)},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    assert reference.success, reference.message
    assert diagnostics["solver_converged"]
    assert alpha.sum() == pytest.approx(1.0, abs=1e-12)
    assert alpha.min() >= 0
    assert alpha.max() <= cap + 1e-10
    assert alpha @ kernel @ alpha == pytest.approx(reference.fun, abs=1e-7)

    gradient = 2 * kernel @ alpha
    free = (alpha > 1e-7) & (alpha < cap - 1e-7)
    assert free.any()
    multiplier = gradient[free].mean()
    # Interior gradient equality and the two inequality signs are the KKT
    # optimality conditions for a box-constrained simplex convex quadratic.
    assert np.max(np.abs(gradient[free] - multiplier)) < 2e-5
    assert np.max(multiplier - gradient[alpha <= 1e-7], initial=0) < 2e-5
    assert np.max(gradient[alpha >= cap - 1e-7] - multiplier, initial=0) < 2e-5
    if nu == 0.4:
        assert np.count_nonzero(alpha >= cap - 1e-7) > 0


def test_zero_dual_does_not_certify_ordinary_softmax_eviction():
    # All keys coincide, so there is a sparse optimum of the SVDD objective.
    # A zero-dual key still has positive ordinary-softmax attention mass.
    keys = np.ones((1, 10, 2), dtype=np.float32)
    scores, _ = compute_scores(keys, "svdd")
    zero = np.flatnonzero(scores[0] == 0)
    assert len(zero) > 0
    values = np.zeros(10)
    values[zero[0]] = 1
    query = np.zeros(2)
    logits = keys[0] @ query
    weights = np.exp(logits) / np.exp(logits).sum()
    full_output = weights @ values
    retained = np.flatnonzero(scores[0] > 0)
    weights_after = np.exp(logits[retained]) / np.exp(logits[retained]).sum()
    evicted_output = weights_after @ values[retained]
    assert full_output == pytest.approx(0.1)
    assert evicted_output == 0


def test_distinct_key_softmax_counterexample():
    from sklearn.svm import OneClassSVM

    keys = np.array([[-1.0], [0.0], [1.0]])
    fit = OneClassSVM(kernel="rbf", gamma=0.25, nu=0.5, tol=1e-10).fit(keys)
    alpha = np.zeros(3)
    alpha[fit.support_] = fit.dual_coef_[0] / fit.dual_coef_[0].sum()
    np.testing.assert_allclose(alpha, [0.5, 0.0, 0.5], atol=1e-8)
    # q=0 gives equal softmax mass to all three keys, even the zero-alpha key.
    values = np.array([0.0, 1.0, 0.0])
    assert values.mean() == pytest.approx(1 / 3)
    assert values[alpha > 0].mean() == 0


@pytest.mark.parametrize("method", ["svdd", "keydiff", "leverage", "knorm"])
def test_zero_keys_are_finite(method):
    scores, _ = compute_scores(np.zeros((1, 9, 3)), method)
    assert np.isfinite(scores).all()


def test_ties_and_nested_budgets(keys):
    scores = np.zeros(keys.shape[:2])
    small, _ = select_indices(keys, 50, "snapkv", scores=scores)
    large, _ = select_indices(keys, 80, "snapkv", scores=scores)
    np.testing.assert_array_equal(small[0, :18], np.arange(18))
    assert set(small[0]).issubset(large[0])


def test_invalid_inputs(keys):
    with pytest.raises(ValueError, match="budget"):
        select_indices(keys, 302, "random")
    with pytest.raises(ValueError, match="budget"):
        select_indices(keys, 3.1, "random")
    with pytest.raises(ValueError, match="scores"):
        select_indices(keys, 50, "snapkv", scores=np.zeros((2, 300)))
    with pytest.raises(ValueError, match="externally"):
        compute_scores(keys, "snapkv")
    with pytest.raises(ValueError, match="nonempty"):
        compute_scores(np.zeros((0, 3, 2)), "svdd")
    with pytest.raises(ValueError, match="finite"):
        compute_scores(np.full((1, 3, 2), np.nan), "svdd")
