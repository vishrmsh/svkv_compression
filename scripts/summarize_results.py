"""Summarize a small, paired KV-compression pilot without implying benchmark SOTA.

Example:
    python scripts/summarize_results.py results/pilot/predictions.jsonl

Only prediction rows are inputs. This script never imports MLX or loads a model.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np


METHODS = ["svdd", "svdd_prerope", "snapkv", "streaming", "keydiff", "knorm", "random", "leverage"]
LABELS = {
    "full": "Full cache", "svdd": "SVDD post-RoPE", "svdd_prerope": "SVDD pre-RoPE",
    "snapkv": "SnapKV (question hidden)", "streaming": "Initial + recent",
    "keydiff": "KeyDiff", "knorm": "K-norm", "random": "Random", "leverage": "Leverage only",
}
COLORS = {
    "svdd": "#0072B2", "svdd_prerope": "#56B4E9", "snapkv": "#D55E00",
    "streaming": "#999999", "keydiff": "#009E73", "knorm": "#CC79A7",
    "random": "#E69F00", "leverage": "#332288", "full": "#222222",
}
MEASUREMENTS = [
    "actual_retained_fraction", "prefix_tokens", "compressed_kv_tensor_bytes",
    "full_kv_tensor_bytes", "kv_bytes_after_generation", "kv_tensor_byte_ratio",
    "prefill_s", "key_transfer_s", "score_s", "selection_and_compaction_s",
    "compression_overhead_s", "suffix_ttft_s", "decode_s", "decode_tokens_per_s",
    "generation_s", "generated_tokens", "estimated_single_method_total_s",
    "needle_token_retention", "process_rss_bytes", "mlx_peak_bytes_shared_harness",
]


def budget_key(value):
    return round(float(value), 10)


def finite_values(rows, field):
    return [float(r[field]) for r in rows if r.get(field) is not None and math.isfinite(float(r[field]))]


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def load_rows(path):
    """Reject mixed prompts and duplicates that could silently alter a paired mean."""
    rows, seen, identities, identical_duplicates = [], {}, {}, 0
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON; stop the writer or copy a complete snapshot") from exc
        required = ("case_id", "task", "method", "budget_fraction", "score", "prefix_sha256")
        missing = [f for f in required if f not in raw]
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields {missing}")
        if not math.isfinite(float(raw["score"])) or not 0 <= float(raw["score"]) <= 1:
            raise ValueError(f"{path}:{line_number}: score must be finite and between 0 and 1")
        if not 0 < float(raw["budget_fraction"]) <= 1:
            raise ValueError(f"{path}:{line_number}: invalid retention budget")
        identity = (raw["task"], raw["prefix_sha256"], raw.get("nominal_length"))
        old_identity = identities.setdefault(raw["case_id"], identity)
        if identity != old_identity:
            raise ValueError(f"Conflicting task/prefix/length for case {raw['case_id']}")
        key = (raw["case_id"], raw["method"], budget_key(raw["budget_fraction"]))
        if key in seen:
            if seen[key] != raw:
                raise ValueError(f"Conflicting duplicate prediction row: {key}; choose one run explicitly")
            identical_duplicates += 1
            continue
        seen[key] = raw
        row = dict(raw)
        row["budget_fraction"] = budget_key(row["budget_fraction"])
        full_bytes = row.get("full_kv_tensor_bytes", 0)
        row["kv_tensor_byte_ratio"] = row.get("compressed_kv_tensor_bytes", 0) / full_bytes if full_bytes else None
        parts = [row.get(f) for f in ("key_transfer_s", "score_s", "selection_and_compaction_s")]
        row["compression_overhead_s"] = sum(parts) if all(v is not None for v in parts) else None
        # Score computation is reused between budgets by the runner, but each deployment
        # pays the complete score time once. Never divide it by the number of budgets.
        total_parts = [row.get(f) for f in ("prefill_s", "compression_overhead_s", "generation_s")]
        if all(v is not None for v in total_parts):
            recomputed = sum(total_parts)
            recorded = row.get("estimated_single_method_total_s")
            if recorded is not None and not math.isclose(recorded, recomputed, rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"Inconsistent single-method time in row {key}: {recorded} vs {recomputed}")
            row["estimated_single_method_total_s"] = recomputed
        rows.append(row)
    if not rows:
        raise ValueError("No prediction rows found")
    return rows, identical_duplicates


def aggregate(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(f) for f in fields)].append(row)
    result = []
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        group = groups[key]
        scores = finite_values(group, "score")
        item = dict(zip(fields, key))
        item.update(n_cases=len(group), score_mean=float(np.mean(scores)), score_median=float(np.median(scores)))
        for field in MEASUREMENTS:
            values = finite_values(group, field)
            item[field + "_mean"] = mean_or_none(values)
            item[field + "_n"] = len(values)
        result.append(item)
    return result


def is_retrieval(row):
    return row["task"].startswith("niah") or row["task"] == "passage_retrieval_en"


def full_correct_case_ids(rows):
    return {r["case_id"] for r in rows if r["method"] == "full" and is_retrieval(r) and math.isclose(r["score"], 1., abs_tol=1e-12)}


def bootstrap_mean_ci(deltas, reps=2000, seed=20260927):
    values = np.asarray(deltas, dtype=float)
    if not len(values):
        return None, None
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def stable_seed(seed, key):
    digest = hashlib.sha256(json.dumps([seed, key], sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def paired_comparisons(rows, baselines, reps=2000, seed=20260927):
    """Exact case matching, with full at 100% and other comparators budget matched."""
    full_correct = full_correct_case_ids(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[("task", row["task"], None)].append(row)
        if row["task"].startswith("niah"):
            groups[("niah_length", row["task"], row.get("nominal_length"))].append(row)
    result = []
    for group_key, group in sorted(groups.items(), key=lambda p: str(p[0])):
        index = {(r["case_id"], r["method"], r["budget_fraction"]): r for r in group}
        variants = sorted({(r["method"], r["budget_fraction"]) for r in group if r["method"] != "full"})
        for method, budget in variants:
            for baseline in baselines:
                if method == baseline:
                    continue
                baseline_budget = 1. if baseline == "full" else budget
                method_rows = {cid: r for (cid, m, b), r in index.items() if m == method and b == budget}
                baseline_rows = {cid: r for (cid, m, b), r in index.items() if m == baseline and b == baseline_budget}
                if not baseline_rows:
                    continue
                subgroups = ["all"] + (["full_cache_correct"] if is_retrieval(group[0]) else [])
                for subgroup in subgroups:
                    allowed = full_correct if subgroup == "full_cache_correct" else {r["case_id"] for r in group}
                    method_ids, baseline_ids = set(method_rows) & allowed, set(baseline_rows) & allowed
                    matched = sorted(method_ids & baseline_ids)
                    # An empty subgroup remains visible rather than disappearing.
                    a = [method_rows[c]["score"] for c in matched]
                    b = [baseline_rows[c]["score"] for c in matched]
                    deltas = np.asarray(a) - np.asarray(b)
                    key = (*group_key, method, baseline, budget, subgroup)
                    low, high = bootstrap_mean_ci(deltas, reps, stable_seed(seed, key))
                    result.append(dict(
                        group_type=group_key[0], task=group_key[1], nominal_length=group_key[2], subgroup=subgroup,
                        method=method, baseline=baseline, budget_fraction=budget, baseline_budget_fraction=baseline_budget,
                        n_pairs=len(matched), n_method_available=len(method_ids), n_baseline_available=len(baseline_ids),
                        n_method_without_baseline=len(method_ids - baseline_ids), n_baseline_without_method=len(baseline_ids - method_ids),
                        method_score_mean=mean_or_none(a), baseline_score_mean=mean_or_none(b),
                        score_delta_mean=mean_or_none(deltas.tolist()), bootstrap_ci95_low=low, bootstrap_ci95_high=high,
                        wins=int((deltas > 1e-12).sum()), ties=int((np.abs(deltas) <= 1e-12).sum()), losses=int((deltas < -1e-12).sum()),
                        bootstrap_reps=reps, bootstrap_unit="case", bootstrap_seed=seed,
                    ))
    return result


def missing_predictions(rows, expected_methods, budgets):
    present = {(r["case_id"], r["method"], r["budget_fraction"]) for r in rows}
    cases = {r["case_id"]: r for r in rows}
    expected = [("full", 1.)] + [(m, budget_key(b)) for m in expected_methods for b in budgets]
    return [dict(case_id=cid, task=cases[cid]["task"], method=m, budget_fraction=b)
            for cid in sorted(cases) for m, b in expected if (cid, m, b) not in present]


def write_csv(path, rows, empty_fields=None):
    fields = list(rows[0]) if rows else (empty_fields or ["no_rows"])
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(task_aggregate, niah_aggregate, system_aggregate, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    artifacts = []

    def save(fig, stem):
        for extension in ("png", "pdf"):
            filename = f"{stem}.{extension}"
            fig.savefig(out / filename, dpi=180, bbox_inches="tight")
            artifacts.append(filename)
        plt.close(fig)

    def quality_panel(ax, entries, title):
        methods = sorted({r["method"] for r in entries}, key=lambda m: (m != "full", METHODS.index(m) if m in METHODS else 99))
        for method in methods:
            series = sorted([r for r in entries if r["method"] == method], key=lambda r: r["budget_fraction"])
            if method == "full":
                ax.axhline(series[0]["score_mean"], color=COLORS[method], linestyle="--", linewidth=1.3, label=LABELS[method])
            else:
                ax.plot([100*r["budget_fraction"] for r in series], [r["score_mean"] for r in series],
                        marker="o", markersize=4, linewidth=1.6, color=COLORS.get(method), label=LABELS.get(method, method))
        counts = [r["n_cases"] for r in entries]
        n_label = str(counts[0]) if len(set(counts)) == 1 else f"{min(counts)}–{max(counts)}"
        ax.set(title=f"{title} (n={n_label} per cell)", xlabel="Nominal KV retention (%)", ylabel="Mean task score", ylim=(-.03, 1.03))
        ax.grid(alpha=.18)

    tasks = sorted({r["task"] for r in task_aggregate})
    fig, axes = plt.subplots(math.ceil(len(tasks)/2), 2, figsize=(13, 3.7*math.ceil(len(tasks)/2)), squeeze=False)
    for ax, task in zip(axes.flat, tasks):
        quality_panel(ax, [r for r in task_aggregate if r["task"] == task], task)
    for ax in list(axes.flat)[len(tasks):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.02), frameon=False)
    fig.suptitle("Exploratory pilot: context compressed before the question", y=1.01)
    fig.tight_layout(rect=(0, .11, 1, 1))
    save(fig, "quality_by_task")

    if niah_aggregate:
        panels = sorted({(r["task"], r["nominal_length"]) for r in niah_aggregate})
        fig, axes = plt.subplots(math.ceil(len(panels)/2), 2, figsize=(13, 3.5*math.ceil(len(panels)/2)), squeeze=False)
        for ax, (task, length) in zip(axes.flat, panels):
            quality_panel(ax, [r for r in niah_aggregate if r["task"] == task and r["nominal_length"] == length], f"{task} · {length:,} nominal tokens")
        for ax in list(axes.flat)[len(panels):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.02), frameon=False)
        fig.suptitle("Custom NIAH with repeated prose filler — not official RULER", y=1.01)
        fig.tight_layout(rect=(0, .1, 1, 1))
        save(fig, "niah_by_length")

        lengths = sorted({r["nominal_length"] for r in system_aggregate})
        fig, axes = plt.subplots(len(lengths), 2, figsize=(13, 3.8*len(lengths)), squeeze=False)
        for row_axes, length in zip(axes, lengths):
            entries = [r for r in system_aggregate if r["nominal_length"] == length]
            for method in ["full"] + METHODS:
                series = sorted([r for r in entries if r["method"] == method], key=lambda r: r["budget_fraction"])
                for ax, field, scale in [(row_axes[0], "compressed_kv_tensor_bytes_mean", 2**20), (row_axes[1], "estimated_single_method_total_s_mean", 1.)]:
                    valid = [r for r in series if r.get(field) is not None]
                    if valid:
                        ax.plot([r[field]/scale for r in valid], [r["score_mean"] for r in valid], marker="o", linewidth=1.5,
                                color=COLORS.get(method), label=LABELS.get(method, method))
            row_axes[0].set(xlabel="Compacted prefix KV tensors (MiB)", ylabel="Mean custom NIAH score", title=f"{length:,} nominal tokens: tensor bytes")
            row_axes[1].set(xlabel="Estimated single-method total (s)", ylabel="Mean custom NIAH score", title=f"{length:,} nominal tokens: cost includes scoring")
            for ax in row_axes:
                ax.set_ylim(-.03, 1.03); ax.grid(alpha=.18)
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.025), frameon=False)
        fig.suptitle("Quality and measured costs — shared-process RSS is not per-method memory", y=1.01)
        fig.tight_layout(rect=(0, .12, 1, 1))
        save(fig, "niah_quality_cost")
    return artifacts


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def markdown_summary(rows, task_aggregate, niah_aggregate, subgroup_aggregate, paired, missing, out, source, reps, seed, duplicates):
    tasks = sorted({r["task"] for r in rows})
    full_correct = full_correct_case_ids(rows)
    lines = [
        "# Exploratory KV-compression pilot results", "",
        f"Input: `{source}`. **{len({r['case_id'] for r in rows})} cases, {len(rows)} unique prediction rows**, {len(tasks)} tasks. "
        f"Methods present: {', '.join(sorted({r['method'] for r in rows}))}.", "",
        f"Coverage: **{len(missing)} expected rows missing among observed cases**. "
        "This count cannot detect cases absent from the prediction file altogether; compare with the case manifest. "
        f"Identical duplicate lines ignored: {duplicates}. See `missing_predictions.csv`.", "",
        "These are small, exploratory results on a local runtime. Custom NIAH tasks are **not official RULER**. "
        "LongBench rows are a selected small subset. Different task scores are reported separately; a mixed-task average is not a benchmark score.", "",
        "Compression occurs after full-context prefill and **before the downstream question**. SnapKV observes the context tail with the question hidden. "
        "This measures context-only selection and is not a reproduction of question-visible SnapKV or an online streaming policy.", "",
        "## Task scores", "",
        "| Task | Method | Nominal retention | Cases | Mean score | Prefix KV MiB | Compression overhead (s) | Estimated total (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in task_aggregate:
        lines.append(f"| {r['task']} | {LABELS.get(r['method'],r['method'])} | {r['budget_fraction']:.0%} | {r['n_cases']} | "
                     f"{fmt(r['score_mean'])} | {fmt(r['compressed_kv_tensor_bytes_mean']/2**20 if r['compressed_kv_tensor_bytes_mean'] is not None else None,1)} | "
                     f"{fmt(r['compression_overhead_s_mean'])} | {fmt(r['estimated_single_method_total_s_mean'])} |")
    lines += ["", "## Paired SV comparisons", "",
              f"Intervals below use **{reps:,} paired case-bootstrap replicates**, seed `{seed}`, percentile 95% intervals. "
              "The paired sample is the intersection of available case IDs at the same retention budget; full cache is compared at 100%. "
              "Positive deltas favor the SV method. These intervals are exploratory, unadjusted for multiple comparisons, "
              "and can be degenerate with tiny samples. Cases sharing filler, seeds, or source documents are not necessarily independent; "
              "case-level resampling does not remove that dependence.", "",
              "| Task | SV method | Baseline | Retention | Pairs | Score delta | Bootstrap 95% interval | Wins / ties / losses |",
              "|---|---|---|---:|---:|---:|---|---|",
    ]
    highlights = [r for r in paired if r["group_type"] == "task" and r["subgroup"] == "all" and r["method"] in ("svdd", "svdd_prerope")]
    for r in highlights:
        lines.append(f"| {r['task']} | {LABELS.get(r['method'],r['method'])} | {LABELS.get(r['baseline'],r['baseline'])} | "
                     f"{r['budget_fraction']:.0%} | {r['n_pairs']} | {fmt(r['score_delta_mean'])} | "
                     f"[{fmt(r['bootstrap_ci95_low'])}, {fmt(r['bootstrap_ci95_high'])}] | {r['wins']} / {r['ties']} / {r['losses']} |")
    lines += ["", "All method–baseline comparisons, length-specific deltas, pair counts, and missing-pair diagnostics are in `paired_deltas.csv`.",
              "", "## Retrieval cases solved by full cache", "",
              f"There are **{len(full_correct)} retrieval cases** with full-cache score exactly 1. "
              "This subgroup includes custom NIAH and LongBench passage retrieval; partial full-cache multi-needle matches are excluded. "
              "It asks whether compression preserves success where the base model succeeds. It is a conditional, post-selection diagnostic, "
              "not a replacement for all-case accuracy. Missing full-cache rows are not treated as failures.", "",
              "| Task | Method | Retention | Cases | Conditional mean score |",
              "|---|---|---:|---:|---:|"]
    for r in subgroup_aggregate:
        lines.append(f"| {r['task']} | {LABELS.get(r['method'],r['method'])} | {r['budget_fraction']:.0%} | {r['n_cases']} | {fmt(r['score_mean'])} |")
    if not subgroup_aggregate:
        lines.append("| No eligible full-cache-correct retrieval cases | — | — | 0 | — |")
    lines += ["", "## Cost interpretation and artifacts", "",
              "- **Tensor bytes** are actual compacted prefix K/V tensor payloads, before adding question and generation tokens. "
              "`kv_bytes_after_generation` records later growth. These byte counts exclude weights, selector workspaces, allocator padding, "
              "and shared full-cache copies.",
              "- `process_rss_bytes` and `mlx_peak_bytes_shared_harness` describe a process that retains a full reference and shares allocations "
              "between variants. They are retained in CSV for transparency and **must not be described as independent per-method peak memory savings**.",
              "- Compression overhead includes key transfer, the complete score computation, and selection/compaction. "
              "The runner computes scores once and reuses them for three retention budgets; each estimated deployment total pays the complete "
              "score cost once, without dividing it by three. Estimated total is prefill + compression overhead + question/generation time. "
              "It is an accounting estimate assembled from shared measurements, not an isolated end-to-end deployment benchmark.",
              "- Output lengths vary. Decode tokens/s and generation time are not interchangeable, especially for very short answers. "
              "Weights can be quantized while K/V remains unquantized; this pilot is not evidence against KV quantization.",
              "- `aggregate_by_task.csv`: per-task quality, payload sizes, timing, and diagnostic averages with measurement counts.",
              "- `niah_by_length.csv`: custom single/multiple-needle scores split by nominal length.",
              "- `niah_systems_by_length.csv`: equal-case-weight pooled custom NIAH cost/quality by length; inspect component tasks separately.",
              "- `full_cache_correct_retrieval.csv`: the conditional retrieval subgroup.",
              "- `paired_deltas.csv`: all paired deltas and case-bootstrap intervals; `missing_predictions.csv`: expected missing cells.",
              "- `quality_by_task`, `niah_by_length`, and `niah_quality_cost` plots are exported as standalone PNG/PDF where applicable.",
              "- `summary_manifest.json` records source hash, summarizer hash, environment, bootstrap settings, and artifact names.", "",
              "No paper go/no-go verdict is generated automatically. Read raw answers, full-cache performance, selection diagnostics, "
              "and missing coverage alongside these averages.", ""]
    (out / "results_summary.md").write_text("\n".join(lines))


def summarize(path, out, reps=2000, seed=20260927, baselines=None, expected_methods=None, budgets=(.1, .2, .5), plots=True):
    path, out = Path(path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, duplicates = load_rows(path)
    baseline_names = list(baselines) if baselines is not None else ["full"] + sorted({r["method"] for r in rows if r["method"] != "full"})
    expected_methods = list(expected_methods) if expected_methods is not None else METHODS
    task_aggregate = aggregate(rows, ["task", "method", "budget_fraction"])
    niah_rows = [r for r in rows if r["task"].startswith("niah")]
    niah_aggregate = aggregate(niah_rows, ["task", "nominal_length", "method", "budget_fraction"])
    system_aggregate = aggregate(niah_rows, ["nominal_length", "method", "budget_fraction"])
    correct = full_correct_case_ids(rows)
    subgroup_aggregate = aggregate([r for r in rows if r["case_id"] in correct], ["task", "method", "budget_fraction"])
    paired = paired_comparisons(rows, baseline_names, reps, seed)
    missing = missing_predictions(rows, expected_methods, budgets)
    tables = {
        "aggregate_by_task.csv": task_aggregate, "niah_by_length.csv": niah_aggregate,
        "niah_systems_by_length.csv": system_aggregate, "full_cache_correct_retrieval.csv": subgroup_aggregate,
        "paired_deltas.csv": paired, "missing_predictions.csv": missing,
    }
    for name, contents in tables.items():
        write_csv(out / name, contents)
    artifacts = list(tables) + ["results_summary.md", "summary_manifest.json"]
    if plots:
        artifacts += make_plots(task_aggregate, niah_aggregate, system_aggregate, out)
    markdown_summary(rows, task_aggregate, niah_aggregate, subgroup_aggregate, paired, missing, out, str(path), reps, seed, duplicates)
    manifest = dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), python=platform.python_version(),
                    numpy=np.__version__, bootstrap_reps=reps, seed=seed, expected_methods=expected_methods,
                    expected_budgets=list(budgets), baselines=baseline_names, n_rows=len(rows),
                    n_cases=len({r["case_id"] for r in rows}), n_missing_expected_rows=len(missing), artifacts=artifacts)
    if plots:
        import matplotlib
        manifest["matplotlib"] = matplotlib.__version__
    (out / "summary_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="?", default="results/pilot/predictions.jsonl")
    parser.add_argument("--output", help="Defaults to a summary directory beside the predictions")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--baselines", nargs="+", help="Defaults to every observed method, including full")
    parser.add_argument("--expected-methods", nargs="+", default=METHODS)
    parser.add_argument("--budgets", nargs="+", type=float, default=[.1, .2, .5])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_reps <= 0:
        parser.error("--bootstrap-reps must be positive")
    output = args.output or str(Path(args.predictions).parent / "summary")
    result = summarize(args.predictions, output, args.bootstrap_reps, args.seed, args.baselines,
                       args.expected_methods, args.budgets, not args.no_plots)
    print(json.dumps({"output": output, "cases": result["n_cases"], "rows": result["n_rows"],
                      "missing_expected_rows": result["n_missing_expected_rows"]}, indent=2))


if __name__ == "__main__":
    main()
