#!/usr/bin/env python3
"""Validate public closure artifacts without loading any clinical records."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from svkv.redundancy import aggregate, make_synthetic_context, rbf


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(left, right, path="value"):
    if isinstance(right, dict):
        assert set(left) == set(right), f"Schema mismatch: {path}"
        for key in right:
            assert_close(left[key], right[key], f"{path}.{key}")
    elif isinstance(right, (float, int)) and not isinstance(right, bool):
        assert np.isclose(left, right, atol=1e-12, rtol=0), f"Numeric mismatch: {path}"
    elif isinstance(right, list):
        assert len(left) == len(right), f"Length mismatch: {path}"
        for index, (a, b) in enumerate(zip(left, right)):
            assert_close(a, b, f"{path}[{index}]")
    else:
        assert left == right, f"Value mismatch: {path}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("results/redundancy"))
    args = parser.parse_args()
    root = args.output
    metadata = json.loads((root / "metadata.json").read_text())
    synthetic = json.loads((root / "synthetic_summary.json").read_text())
    clinical = json.loads((root / "clinical_summary.json").read_text())
    rows = json.loads((root / "synthetic_rows.json").read_text())
    checks = []
    archive_hashes = {}
    for relative, expected in metadata["code_sha256"].items():
        actual = digest(root / "execution_code" / relative)
        assert actual == expected, f"Archived source mismatch: {relative}"
        archive_hashes[relative] = actual
    archived_config = json.loads((root / "execution_code/configs/redundancy_check.json").read_text())
    assert metadata["protocol"] == archived_config
    checks.append("Executed source, dependency lock, and frozen configuration hashes match archived bytes and metadata.")

    assert len(rows) == 60 and [r["trial"] for r in rows] == list(range(60))
    assert all(row["status"] == "complete" for row in rows)
    rng = np.random.RandomState(0)
    for row in rows:
        keys, labels, centers = make_synthetic_context(rng)
        query_groups = np.repeat(np.arange(12), 2)
        queries = centers[query_groups] + 0.05 * rng.randn(24, 4)
        held = np.arange(1, 24, 2)
        random_selection = np.sort(rng.choice(66, row["k"], replace=False))
        assert row["selections"]["random"] == random_selection.tolist()
        assert row["n"] == 66 and row["budget"] == row["k"] / 66
        assert row["solver"]["support_count"] == row["k"]
        for method, selected in row["selections"].items():
            assert selected == sorted(set(selected))
            assert len(selected) == (66 if method == "full" else row["k"])
            assert min(selected) >= 0 and max(selected) < 66
            selected = np.asarray(selected)
            weights = rbf(queries[held], keys[selected], 1.2, direct=True)
            output = weights @ np.eye(12)[labels[selected]] / weights.sum(axis=1, keepdims=True).clip(1e-12)
            correct = output.argmax(axis=1) == query_groups[held]
            recomputed = {
                "recall_rare": float(correct[query_groups[held] < 6].mean()),
                "recall_all": float(correct.mean()),
                "rare_group_coverage": float(np.isin(np.arange(6), labels[selected]).mean()),
                "all_group_coverage": float(len(np.unique(labels[selected])) / 12),
            }
            assert_close(row["metrics"][method], recomputed)
    expected_synthetic = aggregate(rows, "synthetic context")
    assert_close({k: v for k, v in synthetic.items() if k != "original_reference"}, expected_synthetic)
    checks.append("All 60 synthetic RNG sequences, stored selection budgets, readout scores, aggregate means, and bootstrap intervals independently recompute.")

    synthetic_references = {"svdd": 0.861, "mass_oracle": 0.319, "random": 0.472, "recency": 0.0}
    for method, target in synthetic_references.items():
        assert round(synthetic["methods"][method]["recall_rare"]["mean"], 3) == target
    clinical_references = {
        "svdd": 0.46437441152406944, "density": 0.22537539510281843,
        "random": 0.3326064163901312, "recency": 0.4087777825522535,
    }
    for method, target in clinical_references.items():
        assert clinical["methods"][method]["retain"]["mean"] == target
    assert clinical["mean_budget"] == 0.32260247984516066
    assert clinical["attempts"] == 1465 and clinical["complete"] == 718
    assert clinical["status_counts"] == {"complete": 718, "no_event": 747}
    assert clinical["event_positive_attempts"] == 718
    checks.append("Original synthetic rounded targets and ICU cohort, budget, and all four original retention means exactly match references.")

    for report in [synthetic, clinical]:
        assert sum(report["status_counts"].values()) == report["attempts"]
        assert report["solver_status_counts"] == {"optimal": report["complete"]}
        for method, metrics in report["methods"].items():
            for metric, estimate in metrics.items():
                assert set(estimate) == {"mean", "bootstrap_95_ci", "bootstrap_samples", "seed"}
                assert 0 <= estimate["mean"] <= 1
                lo, hi = estimate["bootstrap_95_ci"]
                assert 0 <= lo <= hi <= 1
                assert estimate["bootstrap_samples"] == 10000 and estimate["seed"] == 0
                paired = report["svdd_minus_comparator"][method][metric]
                assert_close(paired["mean"], report["methods"]["svdd"][metric]["mean"] - estimate["mean"])
                lo, hi = paired["bootstrap_95_ci"]
                assert -1 <= lo <= hi <= 1
    checks.append("Aggregate status totals, all paired mean differences, metric ranges, and declared interval settings are consistent; all 778 solves report optimal.")

    # Explicit schema/inventory limits prevent accidental patient-level output.
    allowed_clinical_keys = {
        "attempts", "complete", "status_counts", "resampling_unit", "independent_unit_assumption",
        "interval_warning", "mean_budget", "mean_tokens", "mean_retained_tokens", "solver_status_counts",
        "maximum_qp_sum_residual", "methods", "svdd_minus_comparator", "event_positive_attempts", "original_reference",
    }
    assert set(clinical) == allowed_clinical_keys
    assert set(metadata["clinical_cache"]) == {"basename", "sha256"}
    allowed_files = {"metadata.json", "synthetic_rows.json", "synthetic_summary.json", "clinical_summary.json", "summary.csv", "summary.md", "validation.json"}
    if (root / "plot_manifest.json").exists():
        plot = json.loads((root / "plot_manifest.json").read_text())
        assert set(plot["input_sha256"]) == {"synthetic_summary.json", "clinical_summary.json"}
        for name, expected in plot["input_sha256"].items():
            assert digest(root / name) == expected
        assert plot["artifacts"] == ["paired_contrasts.png", "paired_contrasts.pdf"]
        assert plot["data_scope"] == "aggregate summaries only; no patient-level inputs"
        assert digest(Path(__file__).with_name("plot_redundancy.py")) == plot["script_sha256"]
        allowed_files.update(["plot_manifest.json", *plot["artifacts"]])
    allowed_files.update("execution_code/" + path for path in metadata["code_sha256"])
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    assert actual_files <= allowed_files, f"Unreviewed output files: {actual_files - allowed_files}"
    inventory = [{"path": relative, "bytes": (root / relative).stat().st_size,
                  "sha256": digest(root / relative)} for relative in sorted(actual_files - {"validation.json"})]
    checks.append("Clinical JSON is restricted to aggregate scalar summaries/intervals; cache provenance is basename+hash only; output inventory has no patient-level file.")

    assert synthetic["methods"]["svdd"]["rare_group_coverage"]["mean"] == 1
    assert synthetic["methods"]["kernel_leverage"]["rare_group_coverage"]["mean"] == 1
    assert synthetic["methods"]["svdd"]["all_group_coverage"]["mean"] == 1
    assert synthetic["methods"]["kernel_leverage"]["all_group_coverage"]["mean"] == 1
    report = {
        "status": "passed", "validated_utc": datetime.now(timezone.utc).isoformat(),
        "checker": "scripts/validate_redundancy_results.py", "checker_sha256": digest(Path(__file__)),
        "reproduce": "python scripts/validate_redundancy_results.py results/redundancy",
        "checks": checks, "optimal_solves": 778,
        "original_target_reproduction": {"synthetic_rounded_recall": synthetic_references,
                                          "clinical_retention_exact": clinical_references,
                                          "clinical_completed_stays": 718, "clinical_no_event_stays": 747},
        "archived_source_sha256": archive_hashes,
        "interpretation_notes": [
            "SVDD and kernel leverage both retain every singleton group and every group in all 60 synthetic trials. Their 0.025 recall difference reflects retained-set readout/composition, not lost-group coverage.",
            "Full-context synthetic recall is 0.722222, below SVDD 0.861111: the full method is an uncompressed reference, not an accuracy ceiling.",
            "The ICU linear-leverage paired retention interval barely excludes zero and all intervals are descriptive, unadjusted, and assume independent stays.",
        ],
        "validation_limits": [
            "No clinical features, labels, identifiers, per-stay metrics, or private cache were loaded by this checker.",
            "Clinical bootstrap draws, individual selections, and means cannot be independently reconstructed from public aggregates alone; the executable protocol and local input hash support credentialed reproduction.",
            "Synthetic selector optima were not rerun; selected indices and their readout scores were validated. Solver status is the recorded status.",
        ],
        "public_output_inventory": inventory,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "optimal_solves": 778, "files_checked": len(inventory)}))


if __name__ == "__main__":
    main()
