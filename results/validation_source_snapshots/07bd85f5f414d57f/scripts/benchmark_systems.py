"""One method/case per fresh process, with full-cache release before decoding.

Example (run each invocation separately, without another accelerator workload):
  .venv/bin/python scripts/benchmark_systems.py --method svdd --fraction .2

This is a post-prefill compression benchmark. Peak memory includes the original
full prefill, selector temporaries, and the overlap while compacting the cache.
All arms use reserved append buffers for prefill and decoding; the quality
harness's concatenate-based decoder is a separate runtime.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mlx.core as mx
import numpy as np
import psutil
from mlx_lm import load

from svkv.geometry import recover_pre_rope
from svkv.runtime import cache_bytes, fork_cache, install_capture, snapkv_scores
from svkv.selectors import METHODS, compute_scores, select_indices
from svkv.systems_cache import allocated_cache_bytes, prefill_reserved, reserve_selected_cache
from svkv.tasks import digest


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def snapshot_sources(output_directory):
    paths = [Path(__file__).resolve()] + sorted(Path("src/svkv").glob("*.py"))
    records = {str(path.relative_to(Path.cwd()) if path.is_absolute() else path):
               file_sha256(path) for path in paths}
    identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    destination = output_directory / "source_snapshots" / identity[:16]
    for path in paths:
        relative = path.relative_to(Path.cwd()) if path.is_absolute() else path
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError(f"Source snapshot collision: {target}")
        target.write_bytes(content)
    (destination / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    return str(destination), records


def memory_record():
    mx.synchronize()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes, Linux reports KiB. MLX execution is Darwin-only here.
    if platform.system() != "Darwin":
        peak_rss *= 1024
    return {
        "mlx_active_bytes": int(mx.get_active_memory()),
        "mlx_allocator_cache_bytes": int(mx.get_cache_memory()),
        "mlx_peak_active_bytes_since_reset": int(mx.get_peak_memory()),
        "process_rss_bytes": int(psutil.Process().memory_info().rss),
        "process_peak_rss_bytes_lifetime": int(peak_rss),
    }


def clear_capture(model):
    for layer in model.layers:
        layer.self_attn.capture = False
        layer.self_attn.last_queries = None


def compact_single_method(model, cache, method, fraction, seed):
    """Keep CPU arrays local so no original arrays escape into decoding."""
    if method == "full":
        return cache, {"transfer_s": 0., "score_s": 0., "select_and_gather_s": 0.,
                       "layers": [], "tokens_per_head": cache[0].size()}
    budget = int(cache[0].size() * fraction)
    if budget < 1:
        raise ValueError("Requested fraction leaves an empty cache")
    work, details = [], []
    transfer_s = score_s = gather_s = 0.
    observation_scores = None
    if method == "snapkv":
        started = time.perf_counter()
        observation_scores = snapkv_scores(model, cache)
        score_s = time.perf_counter() - started
    for layer_index, original in enumerate(cache):
        selector_name = "svdd" if method == "svdd_prerope" else method
        layer_seed = seed + layer_index * 1009
        if method in ("snapkv", "streaming"):
            # The supplied-score selection API only uses key shape/finiteness.
            # Avoid a needless full-dimensional GPU-to-CPU key copy for these arms.
            keys = np.zeros((original.keys.shape[1], original.size(), 1), dtype=np.float32)
            scores = observation_scores[layer_index] if method == "snapkv" else None
            diagnostics = {"method": method, "cpu_key_transfer": False}
        else:
            started = time.perf_counter()
            keys = np.array(original.keys[0].astype(mx.float32))
            transfer_s += time.perf_counter() - started
            started = time.perf_counter()
            geometry = recover_pre_rope(keys, theta=model.args.rope_theta) if method in (
                "svdd_prerope", "leverage"
            ) else keys
            scores, diagnostics = compute_scores(geometry, selector_name, seed=layer_seed)
            score_s += time.perf_counter() - started
        started = time.perf_counter()
        indices, selected = select_indices(keys, budget, selector_name, scores=scores,
                                           seed=layer_seed)
        work.append(fork_cache([original], [indices])[0])
        gather_s += time.perf_counter() - started
        details.append({"layer": layer_index, "scoring": diagnostics, "selection": selected})
    return work, {"transfer_s": transfer_s, "score_s": score_s,
                  "select_and_gather_s": gather_s, "layers": details,
                  "tokens_per_head": budget}


def fixed_decode(model, suffix, cache, decode_steps):
    """One suffix forward, then exactly decode_steps one-token forwards.

    Greedy tokens feed the next step even after EOS. This is a fixed compute
    workload, not an answer-quality endpoint or an EOS-terminated generation.
    """
    mx.synchronize()
    started = time.perf_counter()
    logits = model(mx.array(suffix)[None], cache=cache)[:, -1]
    token = int(mx.argmax(logits, axis=-1).item())
    ttft_s = time.perf_counter() - started
    produced = [token]
    started = time.perf_counter()
    for _ in range(decode_steps):
        logits = model(mx.array([[token]]), cache=cache)[:, -1]
        token = int(mx.argmax(logits, axis=-1).item())
        produced.append(token)
    decode_s = time.perf_counter() - started
    mx.synchronize()
    return {
        "suffix_ttft_s": ttft_s,
        "decode_forward_steps": decode_steps,
        "decode_s": decode_s,
        "decode_forwards_per_s": decode_steps / decode_s,
        "generated_token_count_including_suffix_prediction": len(produced),
        "generated_token_ids": produced,
        "eos_stopping": False,
        "suffix_plus_decode_s": ttft_s + decode_s,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("full", "svdd_prerope", *METHODS), required=True)
    parser.add_argument("--fraction", type=float, default=.2)
    parser.add_argument("--model", default="models/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--cases", default="data/pilot_cases.jsonl")
    parser.add_argument("--case-id")
    parser.add_argument("--output-directory", default="results/systems")
    parser.add_argument("--output", help="Optional explicit JSON result filename")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--skip-weight-hash", action="store_true",
                        help="Record weight sizes only; default hashes weights before timing")
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or args.decode_steps < 1 or args.chunk_size < 1:
        parser.error("fraction must be in (0,1]; decode steps and chunk size must be positive")
    fraction = 1. if args.method == "full" else args.fraction
    cases = [json.loads(line) for line in Path(args.cases).read_text().splitlines() if line.strip()]
    if args.case_id:
        matches = [case for case in cases if case["id"] == args.case_id]
    else:
        matches = [case for case in cases if case["task"] == "niah_single"
                   and case["nominal_length"] == 32768]
    if not matches:
        parser.error("No requested case (or default 32k single needle) exists")
    case = matches[0]
    if digest(case["prefix"]) != case["prefix_sha256"]:
        raise ValueError("Case prefix hash does not match its manifest")
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    source_directory, source_hashes = snapshot_sources(output_directory)
    model_files = {}
    for path in sorted(Path(args.model).iterdir()):
        if not path.is_file():
            continue
        record = {"bytes": path.stat().st_size}
        if not args.skip_weight_hash or path.suffix != ".safetensors":
            record["sha256"] = file_sha256(path)
        model_files[path.name] = record
    invocation = {
        "arguments": vars(args), "method": args.method, "budget_fraction": fraction,
        "pid": os.getpid(), "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(), "python": platform.python_version(),
        "system_memory_bytes": psutil.virtual_memory().total,
        "packages": {name: importlib.metadata.version(name) for name in (
            "mlx", "mlx-lm", "numpy", "scipy", "scikit-learn", "psutil"
        )},
        "case_id": case["id"], "prefix_sha256": case["prefix_sha256"],
        "case_file_sha256": file_sha256(args.cases),
        "model_files": model_files, "source_snapshot": source_directory,
        "source_sha256": source_hashes,
        "protocol": "one fresh process, post-prefill selection, original cache released before decode",
        "cache_backend": "reserved_append_prefill_and_decode",
        "cache_capacity_policy": "physical prefix rows + suffix token count + fixed decode forwards",
        "weight_quantization_bits": 4, "kv_quantization": False,
        "warmup": "short prefill, one suffix forward and two one-token forwards",
        "notes": [
            "Peak allocator memory includes full prefill and overlapping old/new caches.",
            "RSS includes allocator and library retention; active MLX bytes are separate.",
            "Timing is one measurement per fresh process; repeat invocations for uncertainty.",
            "Fixed decode continues after EOS and is not an answer-quality evaluation.",
            "All systems arms reserve fixed append capacity; no full-cache concatenation during decode.",
            "Quality-harness concatenate timings are separate and are not systems speedup evidence.",
        ],
    }
    print(f"Loading {args.model}; method={args.method} fraction={fraction}", flush=True)
    started = time.perf_counter()
    model, tokenizer = load(args.model)
    model.eval()
    install_capture(model)
    invocation["model_load_s"] = time.perf_counter() - started
    warm_suffix = tokenizer.encode(" Continue.")
    warm, _ = prefill_reserved(model, tokenizer.encode("This is a short system warmup."),
                               capture=False, additional_tokens=len(warm_suffix) + 2)
    fixed_decode(model, warm_suffix, warm, 2)
    del warm
    clear_capture(model)
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    memory = {"model_and_runtime_after_warmup": memory_record()}
    print(f"Prefill {len(case['prefix'])} tokens", flush=True)
    append_allowance = len(case["suffix"]) + args.decode_steps
    cache, prefill_s = prefill_reserved(model, case["prefix"], chunk_size=args.chunk_size,
                                       capture=args.method == "snapkv",
                                       additional_tokens=append_allowance)
    full_bytes = cache_bytes(cache)
    full_allocated_bytes = allocated_cache_bytes(cache)
    memory["after_full_prefill"] = memory_record()
    started = time.perf_counter()
    selected_cache, compression = compact_single_method(model, cache, args.method, fraction, case["seed"])
    selection_and_compaction_s = time.perf_counter() - started
    started = time.perf_counter()
    if args.method == "full":
        # Reuse the prefill buffers; full cache pays no artificial conversion copy.
        work = selected_cache
        reserve_allocation_s = 0.
    else:
        work = reserve_selected_cache(selected_cache, additional_tokens=append_allowance)
        reserve_allocation_s = time.perf_counter() - started
    compression_s = selection_and_compaction_s + reserve_allocation_s
    memory["after_compaction_and_reserve_before_release"] = memory_record()
    del selected_cache
    del cache
    clear_capture(model)
    gc.collect()
    mx.clear_cache()
    memory["after_original_cache_and_temporaries_released"] = memory_record()
    compressed_bytes = cache_bytes(work)
    allocated_bytes = allocated_cache_bytes(work)
    assert all(c.offset == len(case["prefix"]) for c in work)
    assert all(c.size() == compression["tokens_per_head"] for c in work)
    print(f"Decode with {compressed_bytes / 2**20:.2f} MiB active KV", flush=True)
    timing = fixed_decode(model, case["suffix"], work, args.decode_steps)
    assert allocated_cache_bytes(work) == allocated_bytes
    memory["after_fixed_decode"] = memory_record()
    result = {
        **invocation, "prefix_tokens": len(case["prefix"]), "suffix_tokens": len(case["suffix"]),
        "nominal_length": case["nominal_length"],
        "tokens_per_head_before": len(case["prefix"]),
        "tokens_per_head_after_compression": compression["tokens_per_head"],
        "actual_retained_fraction": compression["tokens_per_head"] / len(case["prefix"]),
        "full_kv_tensor_bytes": full_bytes, "compressed_kv_tensor_bytes": compressed_bytes,
        "full_allocated_kv_tensor_bytes_including_reserve": full_allocated_bytes,
        "allocated_kv_tensor_bytes_including_reserve": allocated_bytes,
        "allocated_kv_tensor_bytes_after_decode": allocated_cache_bytes(work),
        "append_reserve_tokens_per_head": append_allowance,
        "allocated_capacity_tokens_per_head": work[0].capacity,
        "unused_reserve_bytes_at_decode_start": allocated_bytes - compressed_bytes,
        "kv_tensor_bytes_after_decode": cache_bytes(work),
        "prefill_s": prefill_s, "compression_total_s": compression_s,
        "selection_and_compaction_s": selection_and_compaction_s,
        "selected_cache_reserve_allocation_s": reserve_allocation_s,
        "compression": compression, "timing": timing, "memory": memory,
        "prefill_selection_suffix_and_decode_s": prefill_s + compression_s + timing["suffix_plus_decode_s"],
    }
    output = Path(args.output) if args.output else output_directory / (
        f"{case['id']}__{args.method}__{fraction:g}__{time.time_ns()}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}; prefill={prefill_s:.3f}s compression={compression_s:.3f}s "
          f"decode={timing['decode_forwards_per_s']:.2f} forwards/s", flush=True)


if __name__ == "__main__":
    main()
