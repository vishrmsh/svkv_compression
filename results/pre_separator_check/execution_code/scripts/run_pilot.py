"""Run paired context-only cache compression. Resume complete rows after interruption."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import mlx.core as mx
import numpy as np
import psutil
from mlx_lm import load
from svkv.runtime import install_capture, prefill, snapkv_scores, fork_cache, cache_bytes, generate
from svkv.selectors import compute_scores, select_indices
from svkv.tasks import score_answer
from svkv.geometry import recover_pre_rope


def write_jsonl(path, obj):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/Qwen2.5-7B-Instruct-4bit")
    p.add_argument("--cases", default="data/pilot_cases.jsonl")
    p.add_argument("--output", default="results/pilot")
    p.add_argument("--methods", nargs="+", default=["svdd", "svdd_prerope", "snapkv", "streaming", "keydiff", "knorm", "leverage", "random"])
    p.add_argument("--budgets", nargs="+", type=float, default=[.1, .2, .5])
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--limit", type=int)
    p.add_argument("--chunk-size", type=int, default=512)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    hashes = {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path("src/svkv").glob("*.py"))}
    config = vars(a) | {"code_sha256":hashes,"case_file_sha256":hashlib.sha256(Path(a.cases).read_bytes()).hexdigest(),
        "platform":platform.platform(), "packages":{m:importlib.metadata.version(m) for m in ["mlx", "mlx-lm", "numpy", "scipy", "scikit-learn", "transformers"]},
        "selection_protocol":"post-prefill, pre-question; same full prefix for every method",
        "weight_quantization_bits":4, "kv_quantization":False,
        "position_policy":"original absolute RoPE, physical chronological selected rows",
        "run_started_local":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / f"invocation_{time.time_ns()}.json").write_text(json.dumps(config, indent=2) + "\n")
    cases = [json.loads(s) for s in Path(a.cases).read_text().splitlines()]
    if a.ids: cases = [c for c in cases if c["id"] in a.ids]
    if a.limit: cases = cases[:a.limit]
    path = out / "predictions.jsonl"
    done = set()
    if path.exists():
        done = {(r["case_id"],r["method"],r["budget_fraction"] ) for r in map(json.loads,path.read_text().splitlines())}
    print("Loading model", flush=True)
    model, tokenizer = load(a.model)
    install_capture(model)
    # One short warmup, excluded from reported prompt latency.
    warm, _ = prefill(model, tokenizer.encode("This is a short warmup."), capture=False)
    generate(model, tokenizer, tokenizer.encode(" Continue."), warm, max_tokens=2)
    del warm
    for case_index, case in enumerate(cases):
        variants = [("full", 1.)] + [(m,b) for m in a.methods for b in a.budgets]
        variants = [v for v in variants if (case["id"], *v) not in done]
        if not variants: continue
        started = time.perf_counter()
        print(f"CASE {case_index+1}/{len(cases)} {case['id']} prefix={len(case['prefix'])}", flush=True)
        cache, prefill_s = prefill(model, case["prefix"], chunk_size=a.chunk_size)
        full_bytes = cache_bytes(cache)
        print(f"  prefill={prefill_s:.2f}s KV={full_bytes/2**20:.1f}MiB", flush=True)
        key_arrays = None
        method_scores, score_times = {}, {}
        compressed = {m for m,b in variants if m != "full"}
        if compressed:
            transfer_start = time.perf_counter()
            key_arrays = [np.array(c.keys[0].astype(mx.float32)) for c in cache]
            transfer_s = time.perf_counter() - transfer_start
            for method in a.methods:
                if method not in compressed: continue
                t0 = time.perf_counter()
                if method == "snapkv":
                    method_scores[method] = snapkv_scores(model, cache)
                elif method == "streaming":
                    method_scores[method] = [None] * len(cache)
                else:
                    scores, details = [], []
                    for layer_i, keys in enumerate(key_arrays):
                        method_keys = recover_pre_rope(keys, theta=model.args.rope_theta) if method in ("svdd_prerope", "leverage") else keys
                        score_method = "svdd" if method == "svdd_prerope" else method
                        s,d = compute_scores(method_keys, score_method, seed=case["seed"] + layer_i * 1009)
                        scores.append(s); details.append(d)
                    method_scores[method] = scores
                    if method.startswith("svdd"):
                        write_jsonl(out / "solver_diagnostics.jsonl", {"case_id":case["id"],"method":method,"layers":details})
                score_times[method] = time.perf_counter() - t0
                print(f"  score {method}={score_times[method]:.2f}s", flush=True)
        else: transfer_s = 0.
        # Random order after scoring limits systematic decode order confounding.
        random.Random(case["seed"] + case_index).shuffle(variants)
        for method, fraction in variants:
            t0 = time.perf_counter()
            selection_details = []
            if method == "full":
                work = fork_cache(cache)
                budget = len(case["prefix"])
                compact_s = time.perf_counter() - t0
                retained_needle_fraction = 1. if case["needle_spans"] else None
            else:
                budget = int(len(case["prefix"]) * fraction)
                indices = []
                for layer_i, keys in enumerate(key_arrays):
                    score_method = "svdd" if method == "svdd_prerope" else method
                    idx, diag = select_indices(keys, budget, score_method, scores=method_scores[method][layer_i])
                    indices.append(idx); selection_details.append(diag)
                work = fork_cache(cache, indices)
                compact_s = time.perf_counter() - t0
                if case["needle_spans"]:
                    needle_positions = np.concatenate([np.arange(x,y) for x,y in case["needle_spans"]])
                    retained_needle_fraction = float(np.mean([np.isin(needle_positions, h).mean() for ind in indices for h in ind]))
                else: retained_needle_fraction = None
            physical_bytes = cache_bytes(work)
            max_tokens = 32 if case["task"].startswith("niah") else (16 if case["task"] == "passage_retrieval_en" else 128)
            prediction, timing = generate(model, tokenizer, case["suffix"], work, max_tokens=max_tokens)
            row = dict(case_id=case["id"], task=case["task"], method=method, budget_fraction=fraction,
                       actual_retained_fraction=budget / len(case["prefix"]), budget_tokens_per_head=budget,
                       prefix_tokens=len(case["prefix"]), suffix_tokens=len(case["suffix"]), nominal_length=case["nominal_length"],
                       seed=case["seed"], depth=case.get("depth"), prefix_sha256=case["prefix_sha256"],
                       prediction=prediction, answers=case["answers"], score=score_answer(case["task"], prediction, case["answers"]),
                       full_kv_tensor_bytes=full_bytes, compressed_kv_tensor_bytes=physical_bytes,
                       kv_bytes_after_generation=cache_bytes(work), prefill_s=prefill_s,
                       key_transfer_s=transfer_s if method not in ("full", "snapkv", "streaming") else 0.,
                       score_s=score_times.get(method,0.), selection_and_compaction_s=compact_s,
                       selection_diagnostics=selection_details if method.startswith("svdd") else [],
                       needle_token_retention=retained_needle_fraction,
                       process_rss_bytes=psutil.Process().memory_info().rss,
                       mlx_peak_bytes_shared_harness=mx.get_peak_memory(), **timing)
            row["estimated_single_method_total_s"] = row["prefill_s"] + row["key_transfer_s"] + row["score_s"] + compact_s + timing["generation_s"]
            write_jsonl(path, row)
            print(f"  {method:10s} {fraction:.0%} score={row['score']:.3f} gen={timing['generation_s']:.2f}s {prediction[:90]!r}", flush=True)
            del work
        del cache, key_arrays, method_scores
        gc.collect(); mx.clear_cache()
        print(f"  case_total={time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
