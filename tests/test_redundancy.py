"""Small CPU checks of the closure probe's mathematics and privacy boundary."""
import json

import numpy as np
import pytest

from svkv import redundancy as red


def test_exact_leverage_matches_direct_formulas():
    x = np.random.default_rng(7).normal(size=(9, 4))
    gram = red.rbf(x, x, 1.2)
    scores = red.geometric_scores(x, gram)
    z = x - x.mean(axis=0)
    linear = np.diag(z @ np.linalg.inv(z.T @ z + 0.01 * np.eye(4)) @ z.T)
    kernel = np.diag(gram @ np.linalg.inv(gram + 0.01 * np.eye(9)))
    np.testing.assert_allclose(scores["linear_leverage"], linear, atol=1e-13)
    np.testing.assert_allclose(scores["kernel_leverage"], kernel, atol=1e-13)


def test_duplicate_rows_share_leverage_and_zero_keys_are_finite():
    x = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    scores = red.geometric_scores(x, red.rbf(x, x, 1.2))
    np.testing.assert_allclose(scores["linear_leverage"], np.full(4, 1 / 4.01))
    np.testing.assert_allclose(scores["kernel_leverage"][:2], scores["kernel_leverage"][2:])
    zero_scores = red.geometric_scores(np.zeros((5, 3)), np.ones((5, 5)))
    np.testing.assert_allclose(zero_scores["kernel_leverage"], np.full(5, 1 / 5.01))
    assert all(np.isfinite(s).all() for s in zero_scores.values())


def test_original_synthetic_rng_order():
    rng = np.random.RandomState(0)
    keys, labels, centers = red.make_synthetic_context(rng)
    reference = np.random.RandomState(0)
    expected_centers = reference.randn(12, 4) * 1.5
    expected = [expected_centers[group] + 0.15 * reference.randn(4)
                for group in range(12) for _ in range(1 if group < 6 else 10)]
    np.testing.assert_array_equal(centers, expected_centers)
    np.testing.assert_array_equal(keys, expected)
    np.testing.assert_array_equal(np.bincount(labels), [1] * 6 + [10] * 6)
    assert rng.randn() == reference.randn()


def test_topk_new_ties_have_fixed_order():
    np.testing.assert_array_equal(red.topk(np.ones(10), 3), [0, 1, 2])


@pytest.mark.parametrize("protocol", ["synthetic", "icu"])
def test_tiny_svdd_support_matches_known_symmetric_solution(protocol):
    pytest.importorskip("cvxpy")
    x = np.array([[-1.0], [0.0], [1.0]])
    support, diagnostic = red.solve_support(red.rbf(x, x, 2.0), 0.45, protocol)
    np.testing.assert_array_equal(support, [0, 2])
    assert diagnostic["qp_sum_residual"] < 1e-8
    assert diagnostic["status"] == "optimal"


def test_synthetic_shared_budget_and_original_query_draws(monkeypatch):
    monkeypatch.setattr(red, "solve_support", lambda *args: (np.arange(35), {"status": "fixture"}))
    rng = np.random.RandomState(0)
    row = red.synthetic_trial(rng, 0)
    control = np.random.RandomState(0)
    red.make_synthetic_context(control)
    control.randn(24, 4)
    expected_random = np.sort(control.choice(66, 35, replace=False))
    assert all(len(selection) == (66 if name == "full" else 35)
               for name, selection in row["selections"].items())
    np.testing.assert_array_equal(row["selections"]["random"], expected_random)
    assert rng.randn() == control.randn()
    assert set(red.NEW_SELECTORS).issubset(row["metrics"])


def test_clinical_loader_excludes_label_channel_and_ids(tmp_path):
    path = tmp_path / "fixture.npz"
    features = np.array(["hr", "sbp", "dbp", "rr", "spo2", "temp_c"])
    sequence = np.arange(24.0).reshape(4, 6)
    raw = sequence.copy()
    raw[:, 4] = [89, 90, 91, 88]
    np.savez(path, features=features, sequences=sequence[None], raw_sequences=raw[None],
             stay_ids=np.array([123456789]))
    loaded = red.load_icu(path, limit=1)
    np.testing.assert_array_equal(loaded[0][0], sequence[:, [0, 1, 2, 3, 5]])
    np.testing.assert_array_equal(loaded[0][1], [True, False, False, True])
    assert len(loaded[0]) == 2


def test_status_accounting_and_clinical_aggregate_has_no_rows():
    rows = [{"status": "no_event"}, {"status": "solver:RuntimeError"}]
    for i, score in enumerate([0.25, 0.75]):
        rows.append({"status": "complete", "n": 10, "k": 4, "budget": 0.4,
                     "private_sentinel": 999123 + i,
                     "solver": {"status": "optimal", "qp_sum_residual": 1e-12},
                     "metrics": {"svdd": {"retain": score}, "linear_leverage": {"retain": score + 0.1}}})
    summary = red.aggregate(rows, "stay", samples=100)
    assert summary["attempts"] == 4 and summary["complete"] == 2
    assert summary["status_counts"] == {"complete": 2, "no_event": 1, "solver:RuntimeError": 1}
    paired = summary["svdd_minus_comparator"]["linear_leverage"]["retain"]
    assert paired["mean"] == pytest.approx(-0.1)
    np.testing.assert_allclose(paired["bootstrap_95_ci"], [-0.1, -0.1])
    serialized = json.dumps(summary)
    assert "private_sentinel" not in serialized and "999123" not in serialized


def test_bootstrap_chunking_matches_original_draw_order():
    values = np.array([0.1, -0.3, 0.8, 0.5])
    result = red.bootstrap(values, samples=613)
    original = np.random.default_rng(0).choice(values, size=(613, 4), replace=True).mean(axis=1)
    np.testing.assert_array_equal(result["bootstrap_95_ci"], np.quantile(original, [0.025, 0.975]))
