"""Audit completeness, budgets, metrics and provenance without running a model."""
import argparse
from collections import Counter
import hashlib
import gzip
import json
import math
from pathlib import Path
from svkv.tasks import digest, score_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/pilot")
    parser.add_argument("--cases", default="data/pilot_cases.jsonl")
    parser.add_argument("--protocol", default="configs/pilot_protocol.json")
    args = parser.parse_args()
    root = Path(args.results)
    protocol = json.loads(Path(args.protocol).read_text())
    cases = {x["id"]: x for x in map(json.loads, Path(args.cases).read_text().splitlines())}
    rows = [json.loads(x) for x in (root / "predictions.jsonl").read_text().splitlines()]
    methods = [x for x in protocol["methods"] if x != "full"]
    expected = {(c,m,b) for c in cases for m in methods for b in protocol["retained_fractions"]}
    expected |= {(c,"full",1.) for c in cases}
    actual = [(r["case_id"],r["method"],r["budget_fraction"]) for r in rows]
    assert len(set(actual)) == len(actual), "Duplicate prediction arms"
    assert set(actual) == expected, f"Missing {len(expected-set(actual))}; extra {len(set(actual)-expected)} arms"
    token_bytes = 28 * 4 * 128 * 2 * 2  # layers * KVheads * dims * K/V * fp16 bytes
    for case in cases.values():
        assert digest(case["prefix"]) == case["prefix_sha256"]
        limit = 32 if case["task"].startswith("niah") else (16 if case["task"] == "passage_retrieval_en" else 128)
        assert len(case["prefix"]) + len(case["suffix"]) + limit <= 32768
        for start, stop in case["needle_spans"]:
            assert 0 <= start < stop <= len(case["prefix"])
    for row in rows:
        case = cases[row["case_id"]]
        assert row["prefix_sha256"] == case["prefix_sha256"]
        assert row["answers"] == case["answers"]
        expected_score = score_answer(row["task"], row["prediction"], row["answers"])
        assert math.isclose(row["score"], expected_score, abs_tol=1e-12)
        budget = int(len(case["prefix"]) * row["budget_fraction"])
        assert row["budget_tokens_per_head"] == budget
        assert row["full_kv_tensor_bytes"] == token_bytes * len(case["prefix"])
        assert row["compressed_kv_tensor_bytes"] == token_bytes * budget
        assert row["kv_bytes_after_generation"] == token_bytes * (budget + len(case["suffix"]) + row["decode_forward_steps"])
        if row["method"].startswith("svdd"):
            assert len(row["selection_diagnostics"]) == 28
            for d in row["selection_diagnostics"]:
                assert d["tokens_after_per_head"] == budget
                assert d["supports_retained"] + d["zero_score_tokens_retained"] == 4 * budget
    diagnostic_path = root / "solver_diagnostics.jsonl"
    if diagnostic_path.exists():
        diagnostic_text = diagnostic_path.read_text()
    else:
        diagnostic_text = gzip.open(str(diagnostic_path) + ".gz", "rt").read()
    diag_rows = [json.loads(x) for x in diagnostic_text.splitlines()]
    expected_solvers = {(c,m) for c in cases for m in methods if m.startswith("svdd")}
    assert {(r["case_id"],r["method"]) for r in diag_rows} == expected_solvers
    assert len(diag_rows) == len(expected_solvers)
    support, blocks, failed = {}, Counter(), Counter()
    for row in diag_rows:
        assert len(row["layers"]) == 28
        method = row["method"]
        support.setdefault(method, [])
        n = len(cases[row["case_id"]]["prefix"])
        for layer in row["layers"]:
            assert layer["solver_total_blocks"] == 4 * math.ceil(n / 256)
            support[method].append(layer["raw_support_fraction"])
            blocks[method] += layer["solver_total_blocks"]
            failed[method] += layer["solver_unconverged_blocks"]
    invocations = list(root.glob("invocation_*.json"))
    for path in invocations:
        invocation = json.loads(path.read_text())
        for relative, value in invocation["code_sha256"].items():
            snapshot = root / "execution_code" / relative
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == value, f"Code mismatch: {relative}"
        assert hashlib.sha256(Path(args.cases).read_bytes()).hexdigest() == invocation["case_file_sha256"]
    report = {"passed":True,"cases":len(cases),"prediction_rows":len(rows),
              "task_case_counts":dict(Counter(c["task"] for c in cases.values())),
              "checked": ["all expected arms exactly once", "input hashes and model context limits", "all scores recomputed", "exact physical KV byte counts before and after generation", "SVDD support accounting", "solver record completeness", "archived code and input hashes match run invocations"],
              "solver_summary":{m:{"blocks":blocks[m],"unconverged_blocks":failed[m],"mean_layer_support_fraction":sum(v)/len(v)} for m,v in support.items()}}
    (root / "validation.json").write_text(json.dumps(report,indent=2) + "\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
