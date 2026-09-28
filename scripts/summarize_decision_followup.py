"""Audit and summarize the fixed, 36-case SVDD decision-score follow-up on CPU.

The old alpha rows are a separately labeled historical comparator. They are
never pooled with the contemporaneous 360 prediction rows. These reused cases
are exploratory diagnostics, not a held-out evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform

import numpy as np

from svkv.tasks import digest, score_answer

_SPEC = importlib.util.spec_from_file_location(
    "_pilot_summary_helpers", Path(__file__).with_name("summarize_results.py"))
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)

METHODS = ("svdd_decision", "keydiff", "leverage")
BUDGETS = (.1, .2, .5)
LAYERS, HEADS, DIMENSIONS, ELEMENT_BYTES = 28, 4, 128, 2
TOKEN_BYTES = LAYERS * HEADS * DIMENSIONS * 2 * ELEMENT_BYTES
LABELS = {"svdd": "Original SVDD alpha", "svdd_decision": "SVDD decision score",
          "keydiff": "KeyDiff", "leverage": "Leverage", "full": "Full cache"}
COLORS = {"svdd": "#777777", "svdd_decision": "#D55E00", "keydiff": "#009E73",
          "leverage": "#332288", "full": "#222222"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_key(row):
    return row["case_id"], row["method"], round(float(row["budget_fraction"]), 10)


def validate_predictions(rows, cases, expected_case_count=36):
    """Rescore text and check physical KV accounting independently of the runner."""
    require(len(cases) == expected_case_count, f"Expected {expected_case_count} needle cases, got {len(cases)}")
    require(all(c["task"].startswith("niah") for c in cases.values()), "Only NIAH cases belong in this follow-up")
    expected = {(cid, "full", 1.) for cid in cases}
    expected |= {(cid, method, budget) for cid in cases for method in METHODS for budget in BUDGETS}
    actual = [arm_key(row) for row in rows]
    require(len(actual) == len(set(actual)), "Duplicate prediction arms")
    require(set(actual) == expected,
            f"Follow-up coverage mismatch: {len(expected - set(actual))} missing, {len(set(actual) - expected)} extra arms")
    for case in cases.values():
        require(digest(case["prefix"]) == case["prefix_sha256"], f"Invalid prefix hash: {case['id']}")
        require(len(case["prefix"]) + len(case["suffix"]) + 32 <= 32768, f"Context overflow: {case['id']}")
        require(bool(case["answers"]), f"Empty answers: {case['id']}")
        for start, stop in case["needle_spans"]:
            require(0 <= start < stop <= len(case["prefix"]), f"Invalid needle span: {case['id']}")
    for row in rows:
        case = cases[row["case_id"]]
        key = arm_key(row)
        for field in ("task", "prefix_sha256", "answers", "nominal_length", "seed", "depth"):
            require(row.get(field) == case.get(field), f"Case identity mismatch {field}: {key}")
        n, suffix = len(case["prefix"]), len(case["suffix"])
        require(row["prefix_tokens"] == n and row["suffix_tokens"] == suffix, f"Input length mismatch: {key}")
        expected_score = score_answer(case["task"], row["prediction"], case["answers"])
        require(math.isclose(row["score"], expected_score, abs_tol=1e-12), f"Recomputed answer score mismatch: {key}")
        budget = int(n * row["budget_fraction"])
        require(row["budget_tokens_per_head"] == budget, f"Token budget mismatch: {key}")
        require(math.isclose(row["actual_retained_fraction"], budget / n, abs_tol=1e-12), f"Retention fraction mismatch: {key}")
        require(row["full_kv_tensor_bytes"] == TOKEN_BYTES * n, f"Full KV bytes mismatch: {key}")
        require(row["compressed_kv_tensor_bytes"] == TOKEN_BYTES * budget, f"Compacted KV bytes mismatch: {key}")
        steps = row["decode_forward_steps"]
        require(isinstance(steps, int) and 0 <= steps <= 32, f"Invalid decode step count: {key}")
        require(0 <= row["generated_tokens"] <= 32, f"Invalid generated token count: {key}")
        require(row["kv_bytes_after_generation"] == TOKEN_BYTES * (budget + suffix + steps), f"Post-decode KV bytes mismatch: {key}")
        if row["method"] == "svdd_decision":
            details = row["selection_diagnostics"]
            require(len(details) == LAYERS, f"Missing selection diagnostics: {key}")
            for detail in details:
                require(detail["tokens_after_per_head"] == budget, f"Selection budget mismatch: {key}")
                require(detail["tokens_before"] == n, f"Selection input length mismatch: {key}")
                # Positive decision scores mean outside a fitted boundary; they
                # must never be mistaken for positive dual support coefficients.
                require("supports_retained" not in detail and "zero_score_tokens_retained" not in detail,
                        f"Decision-score diagnostics incorrectly use support accounting: {key}")
                require(detail["signed_decision_scores"] and not detail["score_sign_identifies_support"],
                        f"Decision-score interpretation mismatch: {key}")
    return dict(passed=True, cases=len(cases), prediction_rows=len(rows),
                task_case_counts=dict(Counter(c["task"] for c in cases.values())),
                checked=["all expected 36-by-10 arms exactly once", "case identities, prefix hashes and context limits",
                         "all answer scores recomputed", "exact physical KV byte counts before and after generation",
                         "decision selection budgets; no alpha-support interpretation"])


def compare_original(rows, old_rows, cases):
    """Require identical inputs; report, rather than hide, repeated-baseline changes."""
    relevant = [r for r in old_rows if r["case_id"] in cases and r["method"] in ("full", "svdd", "keydiff", "leverage")]
    expected = {(cid, "full", 1.) for cid in cases}
    expected |= {(cid, method, budget) for cid in cases for method in ("svdd", "keydiff", "leverage") for budget in BUDGETS}
    keys = [arm_key(r) for r in relevant]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Original comparator rows are incomplete or duplicated")
    for row in relevant:
        case = cases[row["case_id"]]
        for field in ("task", "prefix_sha256", "answers", "nominal_length", "seed", "depth"):
            require(row.get(field) == case.get(field), f"Original input mismatch {field}: {arm_key(row)}")
        require(row["prefix_tokens"] == len(case["prefix"]) and row["suffix_tokens"] == len(case["suffix"]),
                f"Original input length mismatch: {arm_key(row)}")
        require(math.isclose(row["score"], score_answer(row["task"], row["prediction"], row["answers"]), abs_tol=1e-12),
                f"Original answer score mismatch: {arm_key(row)}")
    old_index = {arm_key(r): r for r in relevant}
    comparisons = []
    for row in rows:
        if row["method"] not in ("full", "keydiff", "leverage"):
            continue
        old = old_index[arm_key(row)]
        comparisons.append(dict(case_id=row["case_id"], task=row["task"], nominal_length=row["nominal_length"],
                                method=row["method"], budget_fraction=row["budget_fraction"],
                                original_score=old["score"], followup_score=row["score"],
                                score_delta=row["score"] - old["score"],
                                score_agrees=math.isclose(row["score"], old["score"], abs_tol=1e-12),
                                prediction_agrees=row["prediction"] == old["prediction"]))
    return [r for r in relevant if r["method"] == "svdd"], comparisons


def validate_solver_rows(solver_rows, cases):
    expected = {(cid, "svdd_decision") for cid in cases}
    keys = [(r["case_id"], r["method"]) for r in solver_rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Incomplete or duplicated solver records")
    totals, failures, supports = 0, 0, []
    raw_supports, free_supports, free_near_boundary, token_coordinates = 0, 0, 0, 0
    max_free_residual, tolerances = 0., []
    for row in solver_rows:
        n = len(cases[row["case_id"]]["prefix"])
        require(len(row["layers"]) == LAYERS, "Solver layer count mismatch")
        for layer in row["layers"]:
            blocks = HEADS * math.ceil(n / 256)
            require(layer["solver_total_blocks"] == blocks, "Solver block count mismatch")
            require(layer["block_size"] == 256 and layer["nu"] == .05 and layer["tol"] == 1e-5
                    and layer["max_iter"] == 10000, "Frozen SVDD fit parameters changed")
            require(not layer["global_svdd_solve"] and not layer["exact_softmax_eviction_certificate"],
                    "Unsupported global-solve or softmax-certificate claim")
            require(layer["global_ranking_across_blocks"], "Decision ranker must rank globally across blocks")
            require(len(layer["per_head"]) == HEADS, "Solver head count mismatch")
            head_failures = 0
            for head in layer["per_head"]:
                require(head["gamma"] > 0 and math.isfinite(head["gamma"]), "Invalid kernel bandwidth")
                position = 0
                for block in head["blocks"]:
                    require(block["start"] == position and block["length"] == min(256, n - position),
                            "Noncontiguous solver blocks")
                    require(0 < block["supports"] <= block["length"], "Invalid raw solver support count")
                    require(0 <= block["free_supports_within_boundary_tolerance"] <= block["free_supports"] <= block["supports"],
                            "Invalid free-support boundary accounting")
                    require(block["boundary_tie_tolerance"] > 0, "Invalid boundary tie tolerance")
                    residual = block["free_support_max_abs_decision_score"]
                    require(residual is None or (math.isfinite(residual) and residual >= 0), "Invalid free-support residual")
                    raw_supports += block["supports"]
                    free_supports += block["free_supports"]
                    free_near_boundary += block["free_supports_within_boundary_tolerance"]
                    token_coordinates += block["length"]
                    max_free_residual = max(max_free_residual, residual or 0.)
                    tolerances.append(block["boundary_tie_tolerance"])
                    position += block["length"]
                    head_failures += int(not block["converged"])
                require(position == n, "Solver blocks do not cover every key")
            require(layer["solver_unconverged_blocks"] == head_failures, "Solver convergence summary mismatch")
            require(layer["solver_converged"] == (head_failures == 0), "Solver convergence flag mismatch")
            totals += blocks
            failures += head_failures
            supports.append(layer["raw_support_fraction"])
    return dict(case_records=len(solver_rows), blocks=totals, unconverged_blocks=failures,
                mean_layer_raw_support_fraction=float(np.mean(supports)),
                raw_support_coordinates=raw_supports, token_coordinates=token_coordinates,
                token_weighted_raw_support_fraction=raw_supports / token_coordinates,
                free_support_coordinates=free_supports,
                free_fraction_of_raw_support_coordinates=free_supports / raw_supports,
                free_supports_within_boundary_tolerance=free_near_boundary,
                free_boundary_tolerance_fraction=free_near_boundary / free_supports if free_supports else None,
                max_abs_free_support_decision_score=max_free_residual,
                min_boundary_tie_tolerance=min(tolerances), max_boundary_tie_tolerance=max(tolerances),
                support_note="Raw fit supports are diagnostic only; positive decision scores are not supports.")


def compare_fit_metadata(solver_rows, old_solver_rows, cases):
    """Compare every recorded fit field exactly, excluding changed scoring fields."""
    old = {r["case_id"]: r for r in old_solver_rows if r["method"] == "svdd" and r["case_id"] in cases}
    require(set(old) == set(cases), "Original post-RoPE fit metadata is incomplete")
    examined, differing, examples, blocks = 0, 0, [], 0

    def compare(left, right, fields, location):
        nonlocal examined, differing
        for field in fields:
            examined += 1
            if left[field] != right[field]:
                differing += 1
                if len(examples) < 20:
                    examples.append(dict(**location, field=field, original=right[field], followup=left[field]))

    for record in solver_rows:
        cid = record["case_id"]
        require(len(record["layers"]) == len(old[cid]["layers"]), "Original/follow-up fit layer count mismatch")
        for layer_i, (layer, old_layer) in enumerate(zip(record["layers"], old[cid]["layers"])):
            location = dict(case_id=cid, layer=layer_i)
            compare(layer, old_layer, ("seed", "block_size", "nu", "tol", "max_iter", "normalization", "gamma_rule"), location)
            require(len(layer["per_head"]) == len(old_layer["per_head"]), "Original/follow-up fit head count mismatch")
            for head_i, (head, old_head) in enumerate(zip(layer["per_head"], old_layer["per_head"])):
                location = dict(case_id=cid, layer=layer_i, head=head_i)
                compare(head, old_head, ("head", "gamma", "gamma_sample_size", "median_positive_squared_distance",
                                         "gamma_fallback_identical_keys", "raw_support_fraction"), location)
                require(len(head["blocks"]) == len(old_head["blocks"]), "Original/follow-up fit block count mismatch")
                for block, old_block in zip(head["blocks"], old_head["blocks"]):
                    blocks += 1
                    location = dict(case_id=cid, layer=layer_i, head=head_i, block_start=block["start"])
                    compare(block, old_block, ("start", "length", "supports", "converged", "iterations", "raw_dual_sum"), location)
    return dict(all_recorded_fit_metadata_agree=differing == 0, blocks_compared=blocks,
                field_values_compared=examined, differing_field_values=differing, first_differences=examples,
                comparison="Exact serialized values for per-layer settings, per-head bandwidths, and per-block fit metadata; excludes score-specific fields.",
                limitation="Agreement checks recorded metadata, not an archived copy of every alpha coefficient or kernel matrix.")


def grouped_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[("all", "all", None)].append(row)
        groups[("task", row["task"], None)].append(row)
        groups[("length", "all", row["nominal_length"])].append(row)
        groups[("task_length", row["task"], row["nominal_length"])].append(row)
    return groups


def aggregate(rows, source):
    out = []
    for (group, task, length), members in sorted(grouped_rows(rows).items(), key=lambda item: str(item[0])):
        arms = defaultdict(list)
        for row in members:
            arms[(row["method"], row["budget_fraction"])].append(row)
        for (method, budget), arm in sorted(arms.items()):
            out.append(dict(source=source, group_type=group, task=task, nominal_length=length,
                            method=method, budget_fraction=budget, n_cases=len(arm),
                            score_mean=float(np.mean([r["score"] for r in arm])),
                            score_s_mean=float(np.mean([r["score_s"] for r in arm])),
                            needle_token_retention_mean=float(np.mean([r["needle_token_retention"] for r in arm]))))
    return out


def paired_deltas(rows, old_alpha, reps=2000, seed=20260927):
    old_index = {arm_key(row): row for row in old_alpha}
    out = []
    for group_key, members in sorted(grouped_rows(rows).items(), key=lambda item: str(item[0])):
        index = {arm_key(row): row for row in members}
        for budget in BUDGETS:
            chosen = sorted([r for r in members if r["method"] == "svdd_decision" and r["budget_fraction"] == budget],
                            key=lambda r: r["case_id"])
            for baseline in ("full", "keydiff", "leverage", "svdd"):
                baseline_budget = 1. if baseline == "full" else budget
                lookup = old_index if baseline == "svdd" else index
                paired = [(r, lookup[(r["case_id"], baseline, baseline_budget)]) for r in chosen]
                values = np.array([a["score"] - b["score"] for a, b in paired])
                low, high = _HELPERS.bootstrap_mean_ci(values, reps, _HELPERS.stable_seed(seed, (*group_key, budget, baseline)))
                out.append(dict(group_type=group_key[0], task=group_key[1], nominal_length=group_key[2],
                                method="svdd_decision", baseline=baseline,
                                baseline_source="original_pilot" if baseline == "svdd" else "followup",
                                budget_fraction=budget, baseline_budget_fraction=baseline_budget, n_pairs=len(paired),
                                decision_score_mean=float(np.mean([a["score"] for a, _ in paired])),
                                baseline_score_mean=float(np.mean([b["score"] for _, b in paired])),
                                score_delta_mean=float(np.mean(values)), bootstrap_ci95_low=low, bootstrap_ci95_high=high,
                                wins=int(np.sum(values > 1e-12)), ties=int(np.sum(np.abs(values) <= 1e-12)),
                                losses=int(np.sum(values < -1e-12)), bootstrap_reps=reps,
                                bootstrap_seed=seed, bootstrap_unit="case"))
    return out


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(aggregates, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True, sharey=True)
    lengths = sorted({r["nominal_length"] for r in aggregates if r["group_type"] == "length"})
    panels = [("all", None)] + [("length", length) for length in lengths]
    for ax, (group_type, length) in zip(axes.flat, panels):
        entries = [r for r in aggregates if r["group_type"] == group_type and r["nominal_length"] == length]
        for method in ("svdd", "svdd_decision", "keydiff", "leverage", "full"):
            series = sorted([r for r in entries if r["method"] == method], key=lambda r: r["budget_fraction"])
            if method == "full":
                ax.axhline(series[0]["score_mean"], color=COLORS[method], ls=":", lw=1.3, label=LABELS[method])
            else:
                ax.plot([100*r["budget_fraction"] for r in series], [r["score_mean"] for r in series],
                        color=COLORS[method], ls="--" if method == "svdd" else "-", marker="o", ms=5,
                        lw=1.8, label=LABELS[method])
        count = entries[0]["n_cases"]
        title = "All reused needle cases" if length is None else f"{length:,} nominal context tokens"
        ax.set(title=f"{title} (n={count})", xlabel="Nominal KV retention (%)", ylabel="Mean numeric recall",
               xticks=[10, 20, 50], ylim=(-.03, 1.04))
        ax.grid(alpha=.18)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .015), frameon=False)
    fig.suptitle("Fixed decision-score follow-up: custom needles, question hidden", y=.98)
    fig.text(.5, .09, "Alpha is from the original pilot; all other lines are contemporaneous reruns. Not held out or official RULER.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .96))
    for extension in ("png", "pdf"):
        fig.savefig(out / f"decision_comparison.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_report(aggregates, pairs, agreement, validation, out, reps, seed):
    lines = ["# Fixed SVDD decision-score follow-up", "",
             f"**{validation['cases']} reused needle cases; {validation['prediction_rows']} new answers.** "
             "The original alpha results are a separate historical comparator. No rows are pooled across runs.", "",
             "This changes ranking from normalized alpha coefficients to signed radius excess while preserving the original block fits. "
             "Both rankers select globally across blocks within each head; neither assigns equal per-block retention quotas. "
             "A blockwise score is not a global SVDD solve. The selector remains query-independent and uses ordinary softmax after physical compaction.", "",
             "These same 36 custom needle cases were inspected in the original pilot. This is a fixed diagnostic follow-up, "
             "**not held-out evidence** and not official RULER. Matching a cheaper baseline cannot establish a reason to reopen the paper.", "",
             "## Mean numeric recall", "",
             "| Method | 10% KV | 20% KV | 50% KV | Cases per budget |", "|---|---:|---:|---:|---:|"]
    overall = [r for r in aggregates if r["group_type"] == "all"]
    for method in ("svdd", "svdd_decision", "keydiff", "leverage"):
        values = sorted([r for r in overall if r["method"] == method], key=lambda r: r["budget_fraction"])
        lines.append(f"| {LABELS[method]} | " + " | ".join(f"{r['score_mean']:.4f}" for r in values) + f" | {values[0]['n_cases']} |")
    full = next(r for r in overall if r["method"] == "full")
    lines += ["", f"Contemporaneous full-cache recall: **{full['score_mean']:.4f}** on {full['n_cases']} cases. "
              "Each multi-needle case receives mean numeric recall across its two gold numbers; extra guesses are not penalized."]
    for grouping, label, field in (("task", "Task", "task"), ("length", "Nominal context tokens", "nominal_length")):
        lines += ["", f"### By {label.lower()}", "",
                  f"| {label} | Method | 10% KV | 20% KV | 50% KV | Cases per budget |",
                  "|---|---|---:|---:|---:|---:|"]
        members = [r for r in aggregates if r["group_type"] == grouping]
        for key in sorted({r[field] for r in members}):
            for method in ("svdd", "svdd_decision", "keydiff", "leverage"):
                values = sorted([r for r in members if r[field] == key and r["method"] == method], key=lambda r: r["budget_fraction"])
                lines.append(f"| {key} | {LABELS[method]} | " + " | ".join(f"{r['score_mean']:.4f}" for r in values)
                             + f" | {values[0]['n_cases']} |")
    lines += ["", "## Paired differences", "",
              f"Decision-score recall minus the comparator, with {reps:,} paired case-bootstrap replicates (seed `{seed}`). "
              "Intervals are exploratory, unadjusted for multiple comparisons, and do not account for dependence among cases sharing filler or seeds.", "",
              "| Comparator | Retention | Mean difference | Bootstrap 95% interval | Wins / ties / losses |", "|---|---:|---:|---|---|"]
    for row in [r for r in pairs if r["group_type"] == "all"]:
        lines.append(f"| {LABELS[row['baseline']]} | {row['budget_fraction']:.0%} | {row['score_delta_mean']:+.4f} | "
                     f"[{row['bootstrap_ci95_low']:+.4f}, {row['bootstrap_ci95_high']:+.4f}] | "
                     f"{row['wins']} / {row['ties']} / {row['losses']} |")
    lines += ["", "## Shared-harness scoring-stage wall time", "",
              "| Nominal context tokens | SVDD decision score (s) | KeyDiff (s) | Leverage (s) | Cases |",
              "|---:|---:|---:|---:|---:|"]
    length_rows = [r for r in aggregates if r["source"] == "followup" and r["group_type"] == "length" and r["budget_fraction"] == .2]
    for length in sorted({r["nominal_length"] for r in length_rows}):
        by_method = {r["method"]: r for r in length_rows if r["nominal_length"] == length}
        lines.append(f"| {length:,} | " + " | ".join(f"{by_method[m]['score_s_mean']:.3f}" for m in METHODS)
                     + f" | {by_method['svdd_decision']['n_cases']} |")
    lines += ["", "The scoring stage runs once per case and is reused across budgets; these are per-case mean stage times. "
              "They include the layer loop and SVDD diagnostic JSON serialization/output, and leverage includes inverse RoPE. "
              "Key transfer and physical compaction are outside this timer. These are measured harness costs, "
              "not isolated deployment timings or pure arithmetic benchmarks."]
    score_equal = sum(r["score_agrees"] for r in agreement)
    text_equal = sum(r["prediction_agrees"] for r in agreement)
    solver = validation["solver_summary"]
    fit = validation["original_fit_metadata_comparison"]
    lines += ["", "## Independent validation", "",
              f"All expected arms are present exactly once; hashes, gold answers, rescored answers, and exact KV byte accounting passed. "
              f"{solver['blocks']:,} block fits were recorded; {solver['unconverged_blocks']:,} did not converge.", "",
              f"Against the original post-RoPE alpha run, {fit['field_values_compared']:,} recorded fit field values "
              f"were compared exactly across {fit['blocks_compared']:,} blocks: **{fit['differing_field_values']:,} differences**. "
              "This includes bandwidths, support counts, iteration counts, and raw dual sums. "
              "These are fit-metadata checks, not a comparison of archived full coefficient arrays.", "",
              f"Repeated full/KeyDiff/leverage arms agree with the original scores in **{score_equal}/{len(agreement)}** comparisons "
              f"and with exact output text in **{text_equal}/{len(agreement)}**. Every comparison is retained in `baseline_agreement.csv`.", "",
              "Raw solver supports describe the fitted dual only. Positive signed decision scores describe points outside a fitted boundary; "
              "they are not support coefficients and are never reported as supports retained.", "",
              f"The fits contain {solver['free_support_coordinates']:,} free support coordinates "
              f"({solver['free_fraction_of_raw_support_coordinates']:.2%} of all positive-dual support coordinates). "
              f"Of those, {solver['free_supports_within_boundary_tolerance']:,} lie within the recorded block boundary tolerance; "
              f"the largest absolute free-support decision score is {solver['max_abs_free_support_decision_score']:.3g}. "
              "Free supports can therefore remain tied at the fitted boundary within solver precision. "
              "This diagnostic does not establish that boundary ties caused a particular retrieval failure.", "",
              "`aggregate_recall.csv` and `paired_deltas.csv` include task, context-length, and combined task-by-length breakdowns. "
              "`validation.json` and `summary_manifest.json` record checks and provenance. The figure is available as PNG and PDF.", "",
              "Selection timings in these answer rows come from the shared quality harness; they are not an isolated deployment latency benchmark. "
              "The original isolated systems measurements do not measure this new decision-score implementation."]
    (out / "decision_followup_summary.md").write_text("\n".join(lines) + "\n")


def summarize(results, original, case_path, output=None, reps=2000, seed=20260927, plots=True):
    results, original, case_path = Path(results), Path(original), Path(case_path)
    out = Path(output) if output else results / "summary"
    cases = {c["id"]: c for c in read_jsonl(case_path) if c["task"].startswith("niah")}
    rows = read_jsonl(results / "predictions.jsonl")
    validation = validate_predictions(rows, cases)
    old_alpha, agreement = compare_original(rows, read_jsonl(original), cases)
    followup_manifest_path = results / "followup_manifest.json"
    followup_manifest = json.loads(followup_manifest_path.read_text())
    require(followup_manifest["case_file_sha256"] == sha256(case_path), "Frozen case-file hash mismatch")
    require(followup_manifest["original_predictions_sha256"] == sha256(original), "Frozen original-prediction hash mismatch")
    require(followup_manifest["asset_manifest_sha256"] == sha256("data/asset_manifest.json"), "Frozen asset-manifest hash mismatch")
    require(set(followup_manifest["case_ids"]) == set(cases) and len(followup_manifest["case_ids"]) == len(cases),
            "Frozen case IDs mismatch")
    require(followup_manifest["expected_answers"] == len(rows), "Frozen expected row count mismatch")
    for relative, expected in followup_manifest["source_sha256"].items():
        require(sha256(results / "execution_code" / relative) == expected, f"Frozen source hash mismatch: {relative}")
    solver_path = results / "solver_diagnostics.jsonl"
    if not solver_path.exists():
        solver_path = Path(str(solver_path) + ".gz")
    solver_rows = read_jsonl(solver_path)
    validation["solver_summary"] = validate_solver_rows(solver_rows, cases)
    original_solver_path = original.parent / "solver_diagnostics.jsonl.gz"
    if not original_solver_path.exists():
        original_solver_path = original.parent / "solver_diagnostics.jsonl"
    validation["original_fit_metadata_comparison"] = compare_fit_metadata(solver_rows, read_jsonl(original_solver_path), cases)
    invocations = sorted(results.glob("invocation_*.json"))
    require(bool(invocations), "Missing run invocation provenance")
    for path in invocations:
        invocation = json.loads(path.read_text())
        require(invocation["case_file_sha256"] == sha256(case_path), "Invocation input file hash mismatch")
        for relative, expected in invocation["code_sha256"].items():
            require(sha256(results / "execution_code" / relative) == expected, f"Archived source hash mismatch: {relative}")
    validation["checked"].append("archived source and original case-file hashes match run invocations")
    validation["baseline_agreement"] = dict(rows=len(agreement), score_matches=sum(r["score_agrees"] for r in agreement),
                                            exact_text_matches=sum(r["prediction_agrees"] for r in agreement))
    aggregates = aggregate(rows, "followup") + aggregate(old_alpha, "original_pilot")
    pairs = paired_deltas(rows, old_alpha, reps, seed)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("aggregate_recall.csv", aggregates), ("paired_deltas.csv", pairs), ("baseline_agreement.csv", agreement)):
        write_csv(out / name, table)
    (out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    markdown_report(aggregates, pairs, agreement, validation, out, reps, seed)
    if plots:
        make_plot(aggregates, out)
    manifest = dict(predictions_sha256=sha256(results / "predictions.jsonl"), original_predictions_sha256=sha256(original),
                    followup_manifest_sha256=sha256(followup_manifest_path),
                    case_file_sha256=sha256(case_path), solver_diagnostics_sha256=sha256(solver_path),
                    solver_diagnostics_file=solver_path.name,
                    original_solver_diagnostics_sha256=sha256(original_solver_path),
                    original_solver_diagnostics_file=original_solver_path.name,
                    script_sha256=sha256(__file__), helper_script_sha256=sha256(Path(__file__).with_name("summarize_results.py")),
                    invocation_sha256={p.name: sha256(p) for p in invocations},
                    n_cases=len(cases), new_prediction_rows=len(rows), separate_original_alpha_rows=len(old_alpha),
                    expected_methods=["full", *METHODS], budgets=list(BUDGETS), bootstrap_reps=reps, bootstrap_seed=seed,
                    python=platform.python_version(), numpy=np.__version__, held_out=False,
                    outputs=sorted({p.name for p in out.iterdir() if p.is_file()} | {"summary_manifest.json"}))
    if plots:
        import matplotlib
        manifest["matplotlib"] = matplotlib.__version__
    (out / "summary_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/decision_followup")
    parser.add_argument("--original", default="results/pilot/predictions.jsonl")
    parser.add_argument("--cases", default="data/pilot_cases.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    require(args.bootstrap_reps > 0, "Bootstrap repetitions must be positive")
    print(json.dumps(summarize(args.results, args.original, args.cases, args.output,
                               args.bootstrap_reps, args.seed, not args.no_plots), indent=2))


if __name__ == "__main__":
    main()
