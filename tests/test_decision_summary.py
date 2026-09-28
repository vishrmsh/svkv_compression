"""Independent failure-oriented checks for the decision follow-up report."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "decision_summary", Path(__file__).parents[1] / "scripts/summarize_decision_followup.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fixture_data():
    cases, rows = {}, []
    for i, length in enumerate((8192, 16384, 32768)):
        cid = f"fixture_{i}"
        prefix = list(range(512 + i))
        case = dict(id=cid, task="niah_multi", prefix=prefix, suffix=[100, 101],
                    prefix_sha256=summary.digest(prefix), nominal_length=length,
                    seed=17, depth=.1, answers=["1234567", "7654321"], needle_spans=[[30, 40]])
        cases[cid] = case
        arms = [("full", 1.)] + [(m, b) for m in summary.METHODS for b in summary.BUDGETS]
        for method, fraction in arms:
            prediction = "1234567" if method == "svdd_decision" else "1234567, 7654321"
            budget = int(len(prefix) * fraction)
            details = [dict(tokens_before=len(prefix), tokens_after_per_head=budget,
                            signed_decision_scores=True, score_sign_identifies_support=False)
                       for _ in range(summary.LAYERS)] if method == "svdd_decision" else []
            rows.append(dict(case_id=cid, task=case["task"], method=method, budget_fraction=fraction,
                             prefix_sha256=case["prefix_sha256"], answers=case["answers"],
                             nominal_length=length, seed=17, depth=.1, prefix_tokens=len(prefix), suffix_tokens=2,
                             prediction=prediction, score=summary.score_answer(case["task"], prediction, case["answers"]),
                             budget_tokens_per_head=budget, actual_retained_fraction=budget / len(prefix),
                             full_kv_tensor_bytes=summary.TOKEN_BYTES * len(prefix),
                             compressed_kv_tensor_bytes=summary.TOKEN_BYTES * budget,
                             kv_bytes_after_generation=summary.TOKEN_BYTES * (budget + 2 + 3),
                             generated_tokens=4, decode_forward_steps=3, selection_diagnostics=details,
                             score_s=.5, needle_token_retention=.5))
    old = [copy.deepcopy(r) | {"method": "svdd" if r["method"] == "svdd_decision" else r["method"]} for r in rows]
    return cases, rows, old


def test_coverage_rescoring_and_exact_physical_bytes():
    cases, rows, _ = fixture_data()
    assert summary.validate_predictions(rows, cases, 3)["prediction_rows"] == 30
    with pytest.raises(ValueError, match="coverage mismatch"):
        summary.validate_predictions(rows[:-1], cases, 3)
    with pytest.raises(ValueError, match="Duplicate"):
        summary.validate_predictions(rows + [rows[0]], cases, 3)
    bad = copy.deepcopy(rows)
    bad[1]["score"] = 1.
    with pytest.raises(ValueError, match="Recomputed answer score"):
        summary.validate_predictions(bad, cases, 3)
    bad = copy.deepcopy(rows)
    bad[1]["kv_bytes_after_generation"] += summary.TOKEN_BYTES
    with pytest.raises(ValueError, match="Post-decode KV bytes"):
        summary.validate_predictions(bad, cases, 3)
    bad = copy.deepcopy(rows)
    bad[1]["selection_diagnostics"][0]["supports_retained"] = 1
    with pytest.raises(ValueError, match="incorrectly use support"):
        summary.validate_predictions(bad, cases, 3)


def test_old_inputs_must_match_and_baseline_disagreement_remains_visible():
    cases, rows, old = fixture_data()
    changed = next(r for r in old if r["method"] == "keydiff")
    changed["prediction"] = "wrong"
    changed["score"] = 0.
    alpha, agreements = summary.compare_original(rows, old, cases)
    assert len(alpha) == 9
    assert len(agreements) == 21
    assert sum(r["score_agrees"] for r in agreements) == 20
    assert next(r for r in agreements if not r["score_agrees"])["score_delta"] == 1.
    changed["prefix_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Original input mismatch"):
        summary.compare_original(rows, old, cases)


def test_paired_deltas_keep_historical_alpha_separate_and_ignore_input_order():
    cases, rows, old = fixture_data()
    alpha, _ = summary.compare_original(rows, old, cases)
    for row in alpha:
        row["score"] = 0.
    pairs = summary.paired_deltas(rows, alpha, reps=100, seed=13)
    assert pairs == summary.paired_deltas(list(reversed(rows)), list(reversed(alpha)), reps=100, seed=13)
    all_pairs = [r for r in pairs if r["group_type"] == "all"]
    assert len(all_pairs) == 12
    for row in all_pairs:
        assert row["n_pairs"] == 3
        expected = .5 if row["baseline"] == "svdd" else -.5
        assert row["score_delta_mean"] == expected
        assert row["bootstrap_ci95_low"] == row["bootstrap_ci95_high"] == expected
        assert row["baseline_source"] == ("original_pilot" if row["baseline"] == "svdd" else "followup")
    aggregate = summary.aggregate(rows, "followup") + summary.aggregate(alpha, "original_pilot")
    assert all(r["n_cases"] == 3 for r in aggregate if r["group_type"] == "all")


def test_solver_audit_detects_missing_block_and_inconsistent_convergence():
    cases, _, _ = fixture_data()
    solver_rows = []
    for cid, case in cases.items():
        n = len(case["prefix"])
        heads = []
        for h in range(summary.HEADS):
            blocks = [dict(start=s, length=min(256, n-s), supports=10, converged=True)
                      for s in range(0, n, 256)]
            # The partial last block can contain fewer than ten keys.
            for block in blocks:
                block["supports"] = min(block["supports"], block["length"])
                block.update(free_supports=block["supports"], free_supports_within_boundary_tolerance=block["supports"],
                             boundary_tie_tolerance=1e-5, free_support_max_abs_decision_score=1e-6)
            heads.append(dict(head=h, gamma=1., blocks=blocks))
        layer = dict(solver_total_blocks=sum(len(h["blocks"]) for h in heads), block_size=256,
                     nu=.05, tol=1e-5, max_iter=10000, global_svdd_solve=False,
                     exact_softmax_eviction_certificate=False, global_ranking_across_blocks=True,
                     per_head=heads, solver_unconverged_blocks=0, solver_converged=True, raw_support_fraction=.1)
        solver_rows.append(dict(case_id=cid, method="svdd_decision", layers=[copy.deepcopy(layer) for _ in range(summary.LAYERS)]))
    report = summary.validate_solver_rows(solver_rows, cases)
    assert report["blocks"] == 28 * 4 * (2 + 3 + 3)
    assert report["unconverged_blocks"] == 0
    assert report["free_fraction_of_raw_support_coordinates"] == 1.
    assert report["free_boundary_tolerance_fraction"] == 1.
    assert report["max_abs_free_support_decision_score"] == 1e-6
    bad = copy.deepcopy(solver_rows)
    bad[0]["layers"][0]["per_head"][0]["blocks"][0]["converged"] = False
    with pytest.raises(ValueError, match="convergence summary mismatch"):
        summary.validate_solver_rows(bad, cases)
    bad = copy.deepcopy(solver_rows)
    bad[0]["layers"][0]["per_head"][0]["blocks"].pop()
    with pytest.raises(ValueError, match="do not cover every key"):
        summary.validate_solver_rows(bad, cases)


def test_plot_exports_historical_and_contemporaneous_comparison(tmp_path):
    cases, rows, old = fixture_data()
    alpha, _ = summary.compare_original(rows, old, cases)
    aggregates = summary.aggregate(rows, "followup") + summary.aggregate(alpha, "original_pilot")
    summary.make_plot(aggregates, tmp_path)
    assert (tmp_path / "decision_comparison.png").stat().st_size > 1000
    assert (tmp_path / "decision_comparison.pdf").stat().st_size > 1000


def test_exact_fit_metadata_comparison_surfaces_changes_without_rounding():
    layer = dict(seed=17, block_size=256, nu=.05, tol=1e-5, max_iter=10000, normalization="L2", gamma_rule="median",
                 per_head=[dict(head=0, gamma=2., gamma_sample_size=256, median_positive_squared_distance=.5,
                                gamma_fallback_identical_keys=False, raw_support_fraction=.1,
                                blocks=[dict(start=0, length=256, supports=25, converged=True, iterations=19, raw_dual_sum=12.8)])])
    old = [dict(case_id="a", method="svdd", layers=[copy.deepcopy(layer)])]
    new = [dict(case_id="a", method="svdd_decision", layers=[copy.deepcopy(layer)])]
    result = summary.compare_fit_metadata(new, old, {"a": {}})
    assert result["all_recorded_fit_metadata_agree"]
    assert result["blocks_compared"] == 1
    new[0]["layers"][0]["per_head"][0]["gamma"] += 1e-12
    result = summary.compare_fit_metadata(new, old, {"a": {}})
    assert not result["all_recorded_fit_metadata_agree"]
    assert result["differing_field_values"] == 1
    assert result["first_differences"][0]["field"] == "gamma"
