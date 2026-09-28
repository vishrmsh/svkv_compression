"""Small fixtures check pairing and accounting, not pilot outcomes."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("summarize_results", Path(__file__).parents[1] / "scripts/summarize_results.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def row(case, method, score, budget=.2, **extra):
    data = dict(case_id=case, task="niah_multi", method=method, budget_fraction=budget,
                nominal_length=8192, prefix_sha256=case, score=score, prefix_tokens=8000,
                full_kv_tensor_bytes=1000, compressed_kv_tensor_bytes=200 if method != "full" else 1000,
                prefill_s=1., key_transfer_s=.1, score_s=3., selection_and_compaction_s=.2, generation_s=.5)
    return data | extra


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_exact_pairing_full_correct_subgroup_and_bootstrap():
    rows = [row("a", "full", 1, 1.), row("b", "full", .5, 1.),
            row("a", "svdd", .5), row("b", "svdd", 1), row("c", "svdd", 0),
            row("a", "keydiff", 0), row("b", "keydiff", .5)]
    paired = summary.paired_comparisons(rows, ["full", "keydiff"], reps=2000, seed=31)
    found = next(r for r in paired if r["group_type"] == "task" and r["method"] == "svdd" and r["baseline"] == "keydiff" and r["subgroup"] == "all")
    assert found["n_pairs"] == 2
    assert found["n_method_without_baseline"] == 1
    assert found["score_delta_mean"] == .5
    assert found["bootstrap_ci95_low"] == found["bootstrap_ci95_high"] == .5
    conditional = next(r for r in paired if r["group_type"] == "task" and r["method"] == "svdd" and r["baseline"] == "full" and r["subgroup"] == "full_cache_correct")
    assert conditional["n_pairs"] == 1
    assert conditional["score_delta_mean"] == -.5
    assert summary.full_correct_case_ids(rows) == {"a"}
    assert paired == summary.paired_comparisons(list(reversed(rows)), ["full", "keydiff"], reps=2000, seed=31)


def test_cost_is_not_amortized_across_retention_budgets(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [row("a", "svdd", 1, b) for b in (.1, .2, .5)])
    rows, count = summary.load_rows(path)
    assert count == 0
    assert all(r["compression_overhead_s"] == pytest.approx(3.3) for r in rows)
    assert all(r["estimated_single_method_total_s"] == pytest.approx(4.8) for r in rows)
    assert all(r["kv_tensor_byte_ratio"] == .2 for r in rows)


def test_reject_conflicting_duplicates_and_prompt_mismatch(tmp_path):
    path = tmp_path / "predictions.jsonl"
    one = row("a", "svdd", 1)
    write_rows(path, [one, one])
    assert summary.load_rows(path)[1] == 1
    write_rows(path, [one, one | {"score": 0}])
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        summary.load_rows(path)
    write_rows(path, [one, row("a", "keydiff", 1, prefix_sha256="different")])
    with pytest.raises(ValueError, match="Conflicting task/prefix"):
        summary.load_rows(path)


def test_fixture_exports_tables_plots_and_coverage(tmp_path):
    source = tmp_path / "predictions.jsonl"
    write_rows(source, [row("a", "full", 1, 1.), row("a", "svdd", .5), row("a", "keydiff", 1)])
    output = tmp_path / "summary"
    result = summary.summarize(source, output, expected_methods=["svdd", "keydiff"], budgets=[.2], plots=True)
    assert result["n_missing_expected_rows"] == 0
    assert (output / "quality_by_task.png").stat().st_size > 1000
    assert (output / "niah_by_length.pdf").stat().st_size > 1000
    assert "before the downstream question" in (output / "results_summary.md").read_text()
    assert "case-bootstrap" in (output / "results_summary.md").read_text()
    assert "score_delta_mean" in (output / "paired_deltas.csv").read_text()
