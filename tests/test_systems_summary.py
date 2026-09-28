"""Small CPU fixtures for complete-suite accounting and memory/timing labels."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("summarize_systems", Path(__file__).parents[1] / "scripts/summarize_systems.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def fixture_run(baseline=1000, active=2000, prefill=2.):
    stage = {"mlx_active_bytes": active, "mlx_peak_active_bytes_since_reset": 6000,
             "process_rss_bytes": 9000, "process_peak_rss_bytes_lifetime": 12000}
    return {
        "method": "full", "budget_fraction": 1., "actual_retained_fraction": 1.,
        "case_id": "fixture_32768", "prefix_sha256": "fixed", "nominal_length": 32768,
        "prefix_tokens": 32512, "suffix_tokens": 23,
        "cache_backend": "reserved_append_prefill_and_decode",
        "memory": {"model_and_runtime_after_warmup": stage | {"mlx_active_bytes": baseline},
                   "after_original_cache_and_temporaries_released": stage,
                   "after_fixed_decode": stage | {"process_rss_bytes": 10000}},
        "timing": {"decode_forward_steps": 32, "eos_stopping": False, "suffix_ttft_s": .5,
                   "decode_s": 2., "decode_forwards_per_s": 16.},
        "allocated_kv_tensor_bytes_after_decode": 1500,
        "allocated_kv_tensor_bytes_including_reserve": 1500,
        "compressed_kv_tensor_bytes": 1000, "unused_reserve_bytes_at_decode_start": 500,
        "prefill_s": prefill, "selection_and_compaction_s": .25,
        "selected_cache_reserve_allocation_s": 0., "compression_total_s": .25,
        "compression": {"transfer_s": 0., "score_s": 0.},
        "prefill_selection_suffix_and_decode_s": prefill + .25 + .5 + 2.,
    }


def prepare(tmp_path):
    source = tmp_path / "suite"
    source.mkdir()
    (source / "suite.json").write_text(json.dumps({"order": [["full", 0], ["full", 1]]}))
    (source / "full_r0.json").write_text(json.dumps(fixture_run()))
    return source


def test_incomplete_suite_writes_nothing(tmp_path):
    source = prepare(tmp_path)
    target = tmp_path / "summary"
    with pytest.raises(ValueError, match="Suite incomplete"):
        summary.summarize(source, target, expected_count=2)
    assert not target.exists()


def test_complete_suite_medians_ranges_and_exclusion(tmp_path):
    source = prepare(tmp_path)
    (source / "full_r1.json").write_text(json.dumps(fixture_run(baseline=1100, active=2400, prefill=4.)))
    (source / "smoke.json").write_text('{"not_a_run": true}')
    (source / "source_snapshots").mkdir()
    (source / "source_snapshots" / "fake_r9.json").write_text('{}')
    target = tmp_path / "summary"
    rows = summary.summarize(source, target, expected_count=2)
    row, = rows
    assert row["repetitions"] == 2
    assert row["prefill_s_median"] == 3.
    assert row["prefill_s_min"] == 2. and row["prefill_s_max"] == 4.
    assert row["postrelease_active_above_baseline_bytes_median"] == 1150
    assert row["peak_process_rss_lifetime_bytes_median"] == 12000
    assert row["postrelease_process_rss_bytes_median"] == 9000
    text = (target / "systems_summary.md").read_text()
    assert "Current RSS" in text and "Peak RSS" in text
    assert "not confidence intervals" in text
    assert "32 one-token forward steps" in text
    assert len((target / "systems_runs.csv").read_text().splitlines()) == 3


def test_reject_mixed_case_and_timing_misaccounting(tmp_path):
    source = prepare(tmp_path)
    second = fixture_run() | {"prefix_sha256": "another_prefix"}
    (source / "full_r1.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="Mixed cases"):
        summary.load_complete_suite(source, expected_count=2)
    second = fixture_run() | {"prefill_selection_suffix_and_decode_s": 999.}
    (source / "full_r1.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="Inconsistent total time"):
        summary.load_complete_suite(source, expected_count=2)
