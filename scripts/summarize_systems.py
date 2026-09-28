"""CPU-only summaries of a complete isolated systems suite.

Reads only top-level METHOD_rREPEAT.json files listed by suite.json. The default
requires all 14 planned files and two repetitions per method before writing any
summary. No MLX imports, model loads, or accelerator work occur here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


RESULT_NAME = re.compile(r"^(?P<method>[a-z0-9_]+)_r(?P<repeat>[0-9]+)\.json$")
METHOD_ORDER = ["full", "svdd", "svdd_prerope", "keydiff", "leverage", "snapkv", "streaming"]
LABELS = {
    "full": "Full cache", "svdd": "SVDD post-RoPE", "svdd_prerope": "SVDD pre-RoPE",
    "keydiff": "KeyDiff", "leverage": "Leverage only", "snapkv": "SnapKV-style",
    "streaming": "Initial + recent",
}


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return value


def flatten_run(raw, filename):
    memory = raw["memory"]
    baseline = memory["model_and_runtime_after_warmup"]
    released = memory["after_original_cache_and_temporaries_released"]
    decoded = memory["after_fixed_decode"]
    timing = raw["timing"]
    if raw.get("validation_only"):
        raise ValueError(f"Validation-only result cannot enter systems summary: {filename}")
    if raw.get("cache_backend") != "reserved_append_prefill_and_decode":
        raise ValueError(f"Unrecognized or mixed cache backend: {filename}")
    if timing["decode_forward_steps"] != 32 or timing["eos_stopping"] is not False:
        raise ValueError(f"Expected 32 fixed forwards without EOS stopping: {filename}")
    if raw["allocated_kv_tensor_bytes_after_decode"] != raw["allocated_kv_tensor_bytes_including_reserve"]:
        raise ValueError(f"Reserved KV capacity changed during decode: {filename}")
    metrics = {
        "baseline_active_mlx_bytes": baseline["mlx_active_bytes"],
        "postrelease_active_mlx_bytes": released["mlx_active_bytes"],
        "postrelease_active_above_baseline_bytes": released["mlx_active_bytes"] - baseline["mlx_active_bytes"],
        "logical_kv_payload_bytes": raw["compressed_kv_tensor_bytes"],
        "reserved_kv_capacity_bytes": raw["allocated_kv_tensor_bytes_including_reserve"],
        "unused_kv_reserve_bytes": raw["unused_reserve_bytes_at_decode_start"],
        "peak_active_mlx_including_prefill_bytes": max(
            stage["mlx_peak_active_bytes_since_reset"] for stage in memory.values()
        ),
        "postrelease_process_rss_bytes": released["process_rss_bytes"],
        "postdecode_process_rss_bytes": decoded["process_rss_bytes"],
        "peak_process_rss_lifetime_bytes": max(
            stage["process_peak_rss_bytes_lifetime"] for stage in memory.values()
        ),
        "prefill_s": raw["prefill_s"],
        "selection_and_compaction_s": raw["selection_and_compaction_s"],
        "selected_cache_reserve_allocation_s": raw["selected_cache_reserve_allocation_s"],
        "selection_construction_total_s": raw["compression_total_s"],
        "key_transfer_s": raw["compression"]["transfer_s"],
        "score_s": raw["compression"]["score_s"],
        "suffix_ttft_s": timing["suffix_ttft_s"],
        "fixed_decode_s": timing["decode_s"],
        "fixed_decode_forwards_per_s": timing["decode_forwards_per_s"],
        "total_prefill_selection_suffix_decode_s": raw["prefill_selection_suffix_and_decode_s"],
    }
    metrics = {name: finite(value, f"{filename}: {name}") for name, value in metrics.items()}
    expected_total = metrics["prefill_s"] + metrics["selection_construction_total_s"] + metrics["suffix_ttft_s"] + metrics["fixed_decode_s"]
    if not math.isclose(metrics["total_prefill_selection_suffix_decode_s"], expected_total, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Inconsistent total time: {filename}")
    if metrics["fixed_decode_s"] <= 0 or not math.isclose(
        metrics["fixed_decode_forwards_per_s"], 32 / metrics["fixed_decode_s"], rel_tol=1e-6
    ):
        raise ValueError(f"Inconsistent fixed-decode throughput: {filename}")
    if metrics["reserved_kv_capacity_bytes"] < metrics["logical_kv_payload_bytes"]:
        raise ValueError(f"Reserved storage is smaller than logical payload: {filename}")
    return {
        "method": raw["method"], "budget_fraction": raw["budget_fraction"],
        "actual_retained_fraction": raw["actual_retained_fraction"],
        "case_id": raw["case_id"], "nominal_length": raw["nominal_length"],
        "prefix_tokens": raw["prefix_tokens"], "suffix_tokens": raw["suffix_tokens"],
        "filename": filename, **metrics,
    }, list(metrics)


def load_complete_suite(directory, expected_count=14, expected_repeats=2):
    directory = Path(directory)
    suite_path = directory / "suite.json"
    if not suite_path.exists():
        raise ValueError("suite.json is missing; no summary written")
    plan = json.loads(suite_path.read_text())
    order = [tuple(item) for item in plan["order"]]
    if len(order) != expected_count or len(set(order)) != len(order):
        raise ValueError(f"Expected {expected_count} unique planned runs; found {len(order)}")
    if any(count != expected_repeats for count in Counter(method for method, _ in order).values()):
        raise ValueError(f"Expected {expected_repeats} planned repetitions per method")
    expected = {f"{method}_r{repeat}.json" for method, repeat in order}
    found = {path.name for path in directory.glob("*.json") if RESULT_NAME.fullmatch(path.name)}
    if missing := expected - found:
        raise ValueError(f"Suite incomplete; missing {sorted(missing)}; no summary written")
    if extra := found - expected:
        raise ValueError(f"Unplanned result files found: {sorted(extra)}")
    runs, metric_names, identity = [], None, None
    hashes = {"suite.json": hashlib.sha256(suite_path.read_bytes()).hexdigest()}
    for filename in sorted(expected):
        path = directory / filename
        raw = json.loads(path.read_text())
        match = RESULT_NAME.fullmatch(filename)
        if raw["method"] != match["method"]:
            raise ValueError(f"Method conflicts with filename: {filename}")
        run_identity = (
            raw["case_id"], raw["prefix_sha256"], raw["nominal_length"], raw["prefix_tokens"],
            raw["suffix_tokens"], json.dumps(raw.get("source_sha256"), sort_keys=True),
            json.dumps(raw.get("packages"), sort_keys=True),
            json.dumps(raw.get("model_files"), sort_keys=True),
        )
        if identity is None:
            identity = run_identity
        elif identity != run_identity:
            raise ValueError("Mixed cases, source versions, environments, or model files in systems suite")
        row, names = flatten_run(raw, filename)
        row["repeat"] = int(match["repeat"])
        metric_names = names
        runs.append(row)
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    if identity[2] != 32768:
        raise ValueError("Expected the planned nominal-32k systems case")
    return runs, metric_names, hashes


def aggregate(runs, metric_names):
    groups = defaultdict(list)
    for row in runs:
        groups[row["method"]].append(row)
    ordering = lambda method: (METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER), method)
    rows = []
    for method in sorted(groups, key=ordering):
        group = groups[method]
        if len({item["budget_fraction"] for item in group}) != 1:
            raise ValueError(f"Mixed budget fractions for method {method}")
        first = group[0]
        row = {key: first[key] for key in (
            "method", "budget_fraction", "actual_retained_fraction", "case_id",
            "nominal_length", "prefix_tokens", "suffix_tokens"
        )}
        row["repetitions"] = len(group)
        for name in metric_names:
            values = [item[name] for item in group]
            row.update({name + "_median": statistics.median(values),
                        name + "_min": min(values), name + "_max": max(values)})
        rows.append(row)
    return rows


def interval(row, name, scale=1., digits=2):
    values = [row[name + suffix] / scale for suffix in ("_median", "_min", "_max")]
    return f"{values[0]:.{digits}f} [{values[1]:.{digits}f}, {values[2]:.{digits}f}]"


def markdown_table(rows):
    case = rows[0]
    lines = [
        "# Isolated systems measurements", "",
        f"One nominal-32k case (`{case['case_id']}`; {case['prefix_tokens']:,} prefix tokens), "
        f"{case['repetitions']} fresh-process repetitions per method. Cells show **median [minimum, maximum]**. "
        "The observed ranges describe these repetitions; they are not confidence intervals.", "",
        "Every arm uses reserved append buffers and 32 one-token forward steps after the same suffix, "
        "continuing after EOS. These measurements describe this Apple/MLX setup; they do not establish CUDA performance. "
        "Quality-harness concatenate timings are a separate measurement.", "",
        "## Memory", "",
        "Active Δ is post-release MLX active memory minus the warmed model/runtime baseline. "
        "Active total includes weights. Logical KV excludes unused reserve; capacity counts backing-array extent "
        "including that reserve. MLX active memory additionally captures allocator alignment. "
        "The MLX peak includes the full prefill and compaction overlap. Current RSS is sampled after releasing "
        "the original cache; peak RSS is the process-lifetime high-water mark, including model loading and warmup.", "",
        "| Method | Active Δ MiB | Active total GiB | Logical KV MiB | KV capacity MiB | Peak MLX GiB | Current RSS GiB | Peak RSS GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fields = [
        ("postrelease_active_above_baseline_bytes", 2**20), ("postrelease_active_mlx_bytes", 2**30),
        ("logical_kv_payload_bytes", 2**20), ("reserved_kv_capacity_bytes", 2**20),
        ("peak_active_mlx_including_prefill_bytes", 2**30), ("postrelease_process_rss_bytes", 2**30),
        ("peak_process_rss_lifetime_bytes", 2**30),
    ]
    for row in rows:
        label = LABELS.get(row["method"], row["method"])
        lines.append("| " + label + " | " + " | ".join(interval(row, field, scale) for field, scale in fields) + " |")
    lines += ["", "## Time", "",
              "Selection includes key export, scoring, ranking/gather, and allocation of the selected cache's "
              "append reserve. Prefill includes allocation of the original reserved cache. Total sums prefill, "
              "selection, suffix TTFT, and the 32 decode forwards; model loading and warmup are excluded. "
              "Decode tokens/s counts one-token forwards, rather than an EOS-terminated answer length.", "",
              "| Method | Prefill s | Selection s | Suffix TTFT s | Decode tokens/s | Total s |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        label = LABELS.get(row["method"], row["method"])
        fields = ["prefill_s", "selection_construction_total_s", "suffix_ttft_s",
                  "fixed_decode_forwards_per_s", "total_prefill_selection_suffix_decode_s"]
        lines.append("| " + label + " | " + " | ".join(interval(row, field, digits=3) for field in fields) + " |")
    lines += ["", "Two repetitions on one context give weak precision and no estimate of workload generalization. "
              "Full-context prefill remains part of every arm's peak-memory cost. "
              "`systems_summary.csv` contains median/min/max for every recorded field; "
              "`systems_runs.csv` preserves each repetition separately.", ""]
    return "\n".join(lines)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory, output, expected_count=14, expected_repeats=2):
    runs, metric_names, hashes = load_complete_suite(directory, expected_count, expected_repeats)
    rows = aggregate(runs, metric_names)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "systems_runs.csv", runs)
    write_csv(output / "systems_summary.csv", rows)
    (output / "systems_summary.md").write_text(markdown_table(rows))
    (output / "systems_summary_manifest.json").write_text(json.dumps({
        "input_directory": str(directory), "source_file_sha256": hashes,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_count": len(runs), "method_count": len(rows), "expected_repetitions": expected_repeats,
        "aggregation": "median, minimum, maximum over fresh-process repetitions; no confidence intervals",
    }, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="results/systems")
    parser.add_argument("--output", default="results/systems_summary")
    parser.add_argument("--expected-count", type=int, default=14)
    parser.add_argument("--expected-repeats", type=int, default=2)
    args = parser.parse_args()
    rows = summarize(args.directory, args.output, args.expected_count, args.expected_repeats)
    print(f"Saved {len(rows)} methods to {args.output}")


if __name__ == "__main__":
    main()
