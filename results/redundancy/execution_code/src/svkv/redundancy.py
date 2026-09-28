"""Frozen SV Attention selection probes with stronger geometric controls.

The data-generation, solver, and scoring protocols follow VyLabs-AI/sv-attention
(Apache-2.0); see docs/redundancy_protocol.md for exact source provenance.
This standalone implementation neither imports nor modifies that repository.
Clinical per-stay results are returned only in memory, never serialized here.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


RIDGE = 0.01
NEW_SELECTORS = ("keydiff", "linear_leverage", "kernel_leverage")


def rbf(a: np.ndarray, b: np.ndarray, width: float, *, direct: bool = False) -> np.ndarray:
    """Original width convention exp(-squared_distance / width**2).

    Preserve the different arithmetic in the original synthetic and ICU paths.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if direct:
        squared = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    else:
        squared = np.maximum(
            np.sum(a * a, axis=1, keepdims=True)
            + np.sum(b * b, axis=1, keepdims=True).T - 2.0 * (a @ b.T), 0.0,
        )
    return np.exp(-squared / width**2)


def geometric_scores(x: np.ndarray, gram: np.ndarray, ridge: float = RIDGE) -> dict[str, np.ndarray]:
    """Exact, query-free controls; no sketch, parameter search, or label access."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("x must be a finite nonempty matrix")
    if gram.shape != (len(x), len(x)) or ridge <= 0:
        raise ValueError("invalid Gram shape or ridge")
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    anchor = normalized.mean(axis=0)
    cosine_x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    anchor = anchor / max(float(np.linalg.norm(anchor)), 1e-8)
    keydiff = -(cosine_x @ anchor)

    centered = x - x.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered + ridge * np.eye(x.shape[1])
    lower = np.linalg.cholesky((covariance + covariance.T) / 2)
    whitened = solve_triangular(lower, centered.T, lower=True, check_finite=False)
    linear = np.einsum("dn,dn->n", whitened, whitened)

    # diag(K (K + lambda I)^-1) = 1 - lambda diag((K + lambda I)^-1).
    # The kernel is intentionally uncentered: it is exactly the SVDD RBF Gram.
    regularized = (gram + gram.T) / 2 + ridge * np.eye(len(x))
    inverse = cho_solve(cho_factor(regularized, lower=True), np.eye(len(x)))
    kernel = 1.0 - ridge * np.diag(inverse)
    return {"keydiff": keydiff, "linear_leverage": linear, "kernel_leverage": kernel}


def topk(scores: np.ndarray, count: int, *, original_sort: bool = False) -> np.ndarray:
    """Preserve old baseline quicksort; new methods break ties by input order."""
    scores = np.asarray(scores)
    if not 0 <= count <= len(scores) or not np.isfinite(scores).all():
        raise ValueError("invalid scores or budget")
    kind = "quicksort" if original_sort else "stable"
    return np.sort(np.argsort(-scores, kind=kind)[:count])


def solve_support(gram: np.ndarray, nu: float, protocol: str) -> tuple[np.ndarray, dict]:
    """Original batch-QP support partition with task-specific tolerances.

    Synthetic performs its original bordered KKT solve after partitioning.
    ICU support membership is fixed before FastOneClassSVM's maintenance-state
    initialization, so the unused inverse/gradient cache is not constructed.
    """
    import cvxpy as cp

    if protocol not in {"synthetic", "icu"} or not 0 < nu <= 1:
        raise ValueError("unknown protocol or infeasible nu")
    n = len(gram)
    cap = 1.0 / (nu * n)
    variable = cp.Variable(n)
    sym = (gram + gram.T) / 2
    objective = cp.quad_form(variable, cp.psd_wrap(sym)) - np.diag(gram) @ variable
    problem = cp.Problem(cp.Minimize(objective), [cp.sum(variable) == 1, variable >= 0, variable <= cap])
    tolerance = 1e-11 if protocol == "synthetic" else 1e-10
    settings = dict(tol_gap_abs=tolerance, tol_gap_rel=tolerance, tol_feas=tolerance)
    if protocol == "icu":
        settings.update(tol_infeas_abs=tolerance, tol_infeas_rel=tolerance)
    problem.solve(solver=cp.CLARABEL, **settings)
    if variable.value is None:
        raise RuntimeError(f"QP did not solve: {problem.status}")
    alpha = np.clip(np.asarray(variable.value, dtype=np.float64), 0.0, cap)
    partition_tolerance = 1e-6 if protocol == "synthetic" else 1e-7
    margin = np.flatnonzero((alpha > partition_tolerance) & (alpha < cap - partition_tolerance))
    capped = np.flatnonzero(alpha >= cap - partition_tolerance)
    support = np.sort(np.concatenate([margin, capped]))
    diagnostic = {
        "status": str(problem.status), "iterations": int(problem.solver_stats.num_iters),
        "partition_tolerance": partition_tolerance, "margin_count": int(len(margin)),
        "capped_count": int(len(capped)), "support_count": int(len(support)),
        "qp_sum_residual": abs(float(alpha.sum()) - 1.0),
    }
    if protocol == "synthetic":
        if not len(margin):
            raise RuntimeError("SVDD: empty margin set")
        bordered = np.zeros((len(margin) + 1, len(margin) + 1))
        bordered[0, 1:] = bordered[1:, 0] = 1
        bordered[1:, 1:] = 2 * gram[np.ix_(margin, margin)]
        target = np.empty(len(margin) + 1)
        target[0] = 1 - len(capped) * cap
        target[1:] = np.diag(gram)[margin] - 2 * cap * gram[np.ix_(margin, capped)].sum(axis=1)
        solution = np.linalg.inv(bordered) @ target
        diagnostic["bordered_max_residual"] = float(np.max(np.abs(bordered @ solution - target)))
    return support, diagnostic


def make_synthetic_context(rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact original RNG order: centers, six singles, sixty near-duplicates."""
    centers = rng.randn(12, 4) * 1.5
    keys, labels = [], []
    for group in range(12):
        for _ in range(1 if group < 6 else 10):
            keys.append(centers[group] + 0.15 * rng.randn(4))
            labels.append(group)
    return np.asarray(keys), np.asarray(labels), centers


def synthetic_trial(rng: np.random.RandomState, index: int) -> dict:
    x, labels, centers = make_synthetic_context(rng)
    gram = rbf(x, x, 1.2, direct=True)
    try:
        support, diagnostic = solve_support(gram, 0.45, "synthetic")
    except Exception as error:
        # Preserve the original stream: failed solves consume no query/random draws.
        return {"trial": index, "status": f"solver:{type(error).__name__}"}
    count, n = len(support), len(x)
    query_groups = np.repeat(np.arange(12), 2)
    queries = centers[query_groups] + 0.05 * rng.randn(len(query_groups), 4)
    warm = np.arange(0, len(queries), 2)
    held = np.arange(1, len(queries), 2)
    similarity = rbf(queries, x, 1.2, direct=True)
    selections = {
        "svdd": support,
        "mass_oracle": topk(similarity.sum(axis=0), count, original_sort=True),
        "mass_warm": topk(similarity[warm].sum(axis=0), count, original_sort=True),
        "recency": np.arange(n - count, n),
        "random": np.sort(rng.choice(n, count, replace=False)),
        "full": np.arange(n),
    }
    scores = geometric_scores(x, gram)
    selections.update({name: topk(score, count) for name, score in scores.items()})
    row = {"trial": index, "status": "complete", "n": n, "k": count,
           "budget": count / n, "solver": diagnostic, "metrics": {}, "selections": {}}
    values = np.eye(12)[labels]
    for name, selected in selections.items():
        weights = similarity[np.ix_(held, selected)]
        output = weights @ values[selected] / weights.sum(axis=1, keepdims=True).clip(1e-12)
        predicted = output.argmax(axis=1)
        correct = predicted == query_groups[held]
        row["metrics"][name] = {
            "recall_rare": float(correct[query_groups[held] < 6].mean()),
            "recall_all": float(correct.mean()),
            "rare_group_coverage": float(np.isin(np.arange(6), labels[selected]).mean()),
            "all_group_coverage": float(len(np.unique(labels[selected])) / 12),
        }
        row["selections"][name] = selected.tolist()
    return row


def load_icu(cache: Path, limit: int = 1465) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read only feature/label arrays from a trusted, credentialed local cache.

    The original archive uses object arrays, hence allow_pickle=True. Never
    load an untrusted archive. Stay IDs are deliberately never accessed.
    """
    with np.load(cache, allow_pickle=True) as archive:
        names = [str(name) for name in archive["features"]]
        if names != ["hr", "sbp", "dbp", "rr", "spo2", "temp_c"]:
            raise ValueError("clinical cache feature order does not match frozen protocol")
        columns = [i for i, name in enumerate(names) if name != "spo2"]
        sequences, raw = archive["sequences"], archive["raw_sequences"]
        if len(sequences) != len(raw) or len(sequences) < limit:
            raise ValueError("clinical cache is smaller than frozen cohort or misaligned")
        return [(np.asarray(sequence)[:, columns], np.asarray(unscaled)[:, 4] < 90)
                for sequence, unscaled in zip(sequences[:limit], raw[:limit])]


def clinical_trial(x: np.ndarray, event: np.ndarray) -> dict:
    """One original held-out-SpO2 case, returned only for in-memory aggregation."""
    event = np.asarray(event, dtype=bool)
    if not event.any():
        return {"status": "no_event"}
    gram = rbf(x, x, 3.0)
    try:
        support, diagnostic = solve_support(gram, 0.3, "icu")
    except Exception as error:
        return {"status": f"solver:{type(error).__name__}"}
    count, n = len(support), len(x)
    if count == 0 or count >= n:
        return {"status": "support_size_boundary", "solver": diagnostic}
    density = gram.copy()
    np.fill_diagonal(density, 0)
    selections = {
        "svdd": support,
        "density": topk(density.sum(axis=0), count, original_sort=True),
        "recency": np.arange(n - count, n),
        "random": np.sort(np.random.RandomState(n).choice(n, count, replace=False)),
        "full": np.arange(n),
    }
    selections.update({name: topk(score, count) for name, score in geometric_scores(x, gram).items()})
    row = {"status": "complete", "budget": count / n, "n": n, "k": count,
           "n_event": int(event.sum()), "solver": diagnostic, "metrics": {}}
    for name, selected in selections.items():
        retained = np.zeros(n, dtype=bool)
        retained[selected] = True
        # Original nearest-retained-key rule (first chronological item wins ties).
        similarity = rbf(np.asarray(x)[event], np.asarray(x)[selected], 3.0)
        nearest = selected[similarity.argmax(axis=1)]
        row["metrics"][name] = {"retain": float(retained[event].mean()),
                                 "retrieval": float(event[nearest].mean())}
    return row


def bootstrap(values: np.ndarray, samples: int = 10_000, seed: int = 0) -> dict:
    """Percentile mean interval; paired differences supplied by the caller."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap needs nonempty finite scalar observations")
    rng = np.random.default_rng(seed)
    # Bound memory without changing the row-major RNG draw sequence.
    draws = np.concatenate([rng.choice(values, size=(min(250, samples - start), len(values)), replace=True).mean(axis=1)
                            for start in range(0, samples, 250)])
    return {"mean": float(values.mean()), "bootstrap_95_ci": np.quantile(draws, [0.025, 0.975]).tolist(),
            "bootstrap_samples": samples, "seed": seed}


def aggregate(rows: list[dict], unit: str, samples: int = 10_000) -> dict:
    """Aggregate scalar outputs only; clinical rows/identifiers never escape."""
    complete = [row for row in rows if row["status"] == "complete"]
    result = {"attempts": len(rows), "complete": len(complete),
              "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
              "resampling_unit": unit, "independent_unit_assumption": True,
              "interval_warning": "Descriptive protocol reuse; no multiplicity adjustment or fresh validation set."}
    if not complete:
        return result
    result["mean_budget"] = float(np.mean([row["budget"] for row in complete]))
    result["mean_tokens"] = float(np.mean([row["n"] for row in complete]))
    result["mean_retained_tokens"] = float(np.mean([row["k"] for row in complete]))
    result["solver_status_counts"] = dict(sorted(Counter(row["solver"]["status"] for row in complete).items()))
    result["maximum_qp_sum_residual"] = max(row["solver"]["qp_sum_residual"] for row in complete)
    methods = list(complete[0]["metrics"])
    metrics = list(complete[0]["metrics"][methods[0]])
    result["methods"], result["svdd_minus_comparator"] = {}, {}
    for method in methods:
        result["methods"][method] = {}
        result["svdd_minus_comparator"][method] = {}
        for metric in metrics:
            values = np.array([row["metrics"][method][metric] for row in complete])
            sv = np.array([row["metrics"]["svdd"][metric] for row in complete])
            result["methods"][method][metric] = bootstrap(values, samples)
            result["svdd_minus_comparator"][method][metric] = bootstrap(sv - values, samples)
    return result
