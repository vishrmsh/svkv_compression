#!/usr/bin/env python3
"""Run frozen legacy selection probes; write aggregate-only clinical evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

from svkv.redundancy import aggregate, clinical_trial, load_icu, synthetic_trial
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ("src/svkv/redundancy.py", "scripts/run_redundancy_check.py",
              "configs/redundancy_check.json", "requirements-redundancy-lock.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "synthetic", "clinical"], default="all")
    parser.add_argument("--clinical-cache", type=Path,
                        help="Trusted, locally credentialed original ICU NPZ; never copied to output")
    parser.add_argument("--output", type=Path, default=ROOT / "results/redundancy")
    args = parser.parse_args()
    if args.task in {"all", "clinical"} and args.clinical_cache is None:
        parser.error("--clinical-cache is required for the clinical probe")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory; prior results are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/redundancy_check.json").read_text())
    config_checks = {
        "synthetic": {"trials": 60, "seed": 0, "nu": 0.45, "rbf_width": 1.2},
        "clinical": {"maximum_stays": 1465, "nu": 0.3, "rbf_width": 3.0},
        "new_baselines": {"ridge": 0.01}, "bootstrap": {"samples": 10000, "seed": 0},
    }
    for section, expected in config_checks.items():
        for key, value in expected.items():
            if config[section][key] != value:
                raise ValueError(f"Frozen protocol changed: {section}.{key}")
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task, "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "cvxpy", "clarabel"]},
        "code_sha256": {}, "protocol": config,
        "clinical_privacy": "Only aggregate summaries written. No features, IDs, or per-stay rows serialized.",
    }
    for name in CODE_FILES:
        source, target = ROOT / name, args.output / "execution_code" / name
        metadata["code_sha256"][name] = sha256(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(args.output / "metadata.json", metadata)
    reports = {}
    start = time.perf_counter()
    if args.task in {"all", "synthetic"}:
        rng = np.random.RandomState(0)
        rows = []
        for trial in range(60):
            rows.append(synthetic_trial(rng, trial))
            if (trial + 1) % 10 == 0:
                print(f"Synthetic attempts complete: {trial + 1}/60", flush=True)
        write_json(args.output / "synthetic_rows.json", rows)
        report = aggregate(rows, "synthetic context")
        report["original_reference"] = {"svdd_recall_rare_rounded": 0.861, "mass_oracle_recall_rare_rounded": 0.319,
                                        "random_recall_rare_rounded": 0.472, "recency_recall_rare_rounded": 0.0}
        reports["synthetic"] = report
        write_json(args.output / "synthetic_summary.json", report)
    if args.task in {"all", "clinical"}:
        metadata["clinical_cache"] = {"basename": args.clinical_cache.name, "sha256": sha256(args.clinical_cache)}
        data = load_icu(args.clinical_cache)
        rows = []
        for index, (features, events) in enumerate(data):
            rows.append(clinical_trial(features, events))
            if (index + 1) % 100 == 0:
                print(f"Clinical attempts complete: {index + 1}/1465", flush=True)
        report = aggregate(rows, "stay")
        report["event_positive_attempts"] = sum(row["status"] != "no_event" for row in rows)
        report["original_reference"] = {"completed_stays": 718, "mean_budget": 0.32260247984516066,
                                        "svdd_retain": 0.46437441152406944, "density_retain": 0.22537539510281843,
                                        "random_retain": 0.3326064163901312, "recency_retain": 0.4087777825522535}
        reports["clinical"] = report
        write_json(args.output / "clinical_summary.json", report)
        # No API below this point accepts clinical rows. Only summaries survive.
        del rows, data, features, events
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["wall_seconds"] = time.perf_counter() - start
    write_json(args.output / "metadata.json", metadata)
    table = []
    for task, report in reports.items():
        for method, metrics in report.get("methods", {}).items():
            for metric, estimate in metrics.items():
                paired = report["svdd_minus_comparator"][method][metric]
                table.append({"task": task, "method": method, "metric": metric, "n": report["complete"],
                              "mean": estimate["mean"], "ci_low": estimate["bootstrap_95_ci"][0],
                              "ci_high": estimate["bootstrap_95_ci"][1], "svdd_minus_method": paired["mean"],
                              "paired_ci_low": paired["bootstrap_95_ci"][0], "paired_ci_high": paired["bootstrap_95_ci"][1]})
    if table:
        with (args.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    lines = ["# Frozen redundancy closure check", "", "The original protocols are reused; these are descriptive comparisons, not a fresh validation set.",
             "Clinical evidence is aggregate only. All non-full methods retain the same count within each context/stay.", ""]
    for task, report in reports.items():
        primary = "recall_rare" if task == "synthetic" else "retain"
        lines += [f"## {task.capitalize()}", "", f"Completed {report['complete']}/{report['attempts']} attempts; statuses `{report['status_counts']}`.",
                  "", "| Method | Mean | SVDD − method | Paired 95% interval |", "|---|---:|---:|---:|"]
        for method, metrics in report.get("methods", {}).items():
            paired = report["svdd_minus_comparator"][method][primary]
            lo, hi = paired["bootstrap_95_ci"]
            lines.append(f"| {method} | {metrics[primary]['mean']:.6f} | {paired['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
        lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote aggregate evidence to {args.output}", flush=True)


if __name__ == "__main__":
    main()
