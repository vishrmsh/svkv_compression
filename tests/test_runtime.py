"""Opt-in Metal integration checks on the actual quantized pilot checkpoint.

Run with SVKV_RUN_METAL_TESTS=1. Imports are deferred so the ordinary CPU test
suite can collect safely on hosts without a usable Metal device.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("SVKV_RUN_METAL_TESTS") != "1",
    reason="Opt in with SVKV_RUN_METAL_TESTS=1 on a Metal-enabled Mac",
)


@pytest.fixture(scope="module")
def runtime_model():
    import mlx.core as mx
    from mlx_lm import load
    from svkv import runtime

    checkpoint = Path(os.environ.get(
        "SVKV_TEST_MODEL", "models/Qwen2.5-7B-Instruct-4bit"
    ))
    if not checkpoint.is_dir():
        pytest.skip(f"Pilot checkpoint absent: {checkpoint}")
    model, tokenizer = load(str(checkpoint))
    model.eval()
    runtime.install_capture(model)
    tokens = tokenizer.encode(
        "The museum catalogue lists the bronze key in cabinet seventeen. " * 40
    )[:333]
    assert len(tokens) == 333
    yield mx, runtime, model, tokenizer, tokens
    for layer in model.layers:
        layer.self_attn.last_queries = None
    mx.clear_cache()


def _fill(mx, model, cache, tokens, chunk=97):
    for start in range(0, len(tokens), chunk):
        model.model(mx.array(tokens[start:start + chunk])[None], cache=cache)
        mx.eval([c.state for c in cache])


def test_full_budget_matches_stock_kvcache_logits(runtime_model):
    """Checks native GQA/RoPE path with both cache implementations."""
    mx, runtime, model, tokenizer, tokens = runtime_model
    from mlx_lm.models.cache import KVCache

    stock = [KVCache() for _ in model.layers]
    compact = [runtime.CompactKVCache() for _ in model.layers]
    _fill(mx, model, stock, tokens)
    _fill(mx, model, compact, tokens)
    indices = [np.tile(np.arange(len(tokens)), (c.keys.shape[1], 1))
               for c in compact]
    compact = runtime.fork_cache(compact, indices)
    suffix = mx.array(tokenizer.encode("Which cabinet contains the bronze key?"))[None]
    native_logits = np.array(model(suffix, cache=stock).astype(mx.float32))
    compact_logits = np.array(model(suffix, cache=compact).astype(mx.float32))
    error = float(np.max(np.abs(native_logits - compact_logits)))
    print(f"full_budget_stock_max_abs_logits={error:.8g}")
    np.testing.assert_allclose(compact_logits, native_logits, rtol=1e-4, atol=1e-4)


def test_head_specific_compaction_preserves_positions_and_bytes(runtime_model):
    mx, runtime, model, _, tokens = runtime_model
    cache, _ = runtime.prefill(model, tokens, chunk_size=128, capture=False)
    n = len(tokens)
    keep = 71
    # Distinct positions per KV head catch accidental broadcasting across heads.
    indices = [np.stack([np.arange(h, h + keep) for h in range(c.keys.shape[1])])
               for c in cache]
    compact = runtime.fork_cache(cache, indices)
    assert runtime.cache_bytes(compact) * n == runtime.cache_bytes(cache) * keep
    for old, new, idx in zip(cache, compact, indices):
        assert old.offset == new.offset == n
        assert new.size() == keep
        assert old.size() == n
        for h in range(old.keys.shape[1]):
            np.testing.assert_array_equal(np.array(new.keys[0, h]),
                                          np.array(old.keys[0, h])[idx[h]])
            np.testing.assert_array_equal(np.array(new.values[0, h]),
                                          np.array(old.values[0, h])[idx[h]])
        np.testing.assert_array_equal(
            np.array(new.make_mask(3, return_array=True)),
            np.arange(keep + 3)[None, :] <= keep + np.arange(3)[:, None],
        )
    print(f"physical_bytes_before={runtime.cache_bytes(cache)} "
          f"after={runtime.cache_bytes(compact)} absolute_offset={n}")
    # Moving the physical length must not change the absolute offset used by RoPE.
    model.model(mx.array([[tokens[-1]]]), cache=compact)
    mx.eval([c.state for c in compact])
    assert all(c.offset == n + 1 and c.size() == keep + 1 for c in compact)
    assert all(c.offset == n and c.size() == n for c in cache)


def test_batched_suffix_matches_sequential_after_compaction(runtime_model):
    """Checks causal masking when physical positions differ from absolute ones."""
    mx, runtime, model, tokenizer, tokens = runtime_model
    from scipy.special import logsumexp
    cache, _ = runtime.prefill(model, tokens, chunk_size=128, capture=False)
    indices = [np.stack([np.arange(h, len(tokens) - 4, 5)
                         for h in range(c.keys.shape[1])]) for c in cache]
    compressed = runtime.fork_cache(cache, indices)
    batched = runtime.fork_cache(compressed)
    sequential = runtime.fork_cache(compressed)
    suffix = tokenizer.encode("The catalogue answer is cabinet seventeen.")
    batch_logits = np.array(model(mx.array(suffix)[None], cache=batched).astype(mx.float32))
    single_logits = np.concatenate([
        np.array(model(mx.array([[token]]), cache=sequential).astype(mx.float32))
        for token in suffix
    ], axis=1)
    error = float(np.max(np.abs(batch_logits - single_logits)))
    batch_lp = batch_logits - logsumexp(batch_logits, axis=-1, keepdims=True)
    single_lp = single_logits - logsumexp(single_logits, axis=-1, keepdims=True)
    max_kl = float(np.max(np.sum(np.exp(batch_lp) * (batch_lp - single_lp), axis=-1)))
    print(f"compressed_batch_vs_sequential_max_abs_logits={error:.8g} max_kl={max_kl:.8g}")
    # Quantized GEMV and GEMM paths use different fp16 arithmetic. A diagnostic
    # native full-cache control also differed (0.0703 max logits, 6.6e-5 max KL).
    # Test distribution agreement and the decision, rather than bitwise equality.
    assert error < 0.25
    assert max_kl < 1e-3
    np.testing.assert_array_equal(batch_logits.argmax(-1), single_logits.argmax(-1))
    assert all(c.offset == len(tokens) + len(suffix) for c in batched + sequential)
    assert all(c.size() == compressed[0].size() + len(suffix) for c in batched + sequential)

    # At a fixed batch shape the fast "causal" path must match an explicit
    # physical-length causal mask exactly, isolating masking from GEMV/GEMM drift.
    explicit = runtime.fork_cache(compressed)
    for c in explicit:
        physical = c.size()
        c.make_mask = lambda n, return_array=False, window_size=None, p=physical: (
            mx.arange(p + n)[None, :] <= (p + mx.arange(n))[:, None]
        )
    explicit_logits = np.array(model(mx.array(suffix)[None], cache=explicit).astype(mx.float32))
    np.testing.assert_array_equal(batch_logits, explicit_logits)

    # Positive control: replacing the absolute RoPE offset by compressed length
    # must be observable. The real runtime preserves the former instead.
    wrong = runtime.fork_cache(compressed)
    for c in wrong:
        c.offset = c.size()
    wrong_logits = np.array(model(mx.array(suffix)[None], cache=wrong).astype(mx.float32))
    assert np.max(np.abs(batch_logits - wrong_logits)) > 1


def test_snapkv_gqa_causal_scores_and_context_only_capture(runtime_model):
    mx, runtime, model, tokenizer, tokens = runtime_model
    from scipy.special import softmax

    window = 32
    # A one-token final chunk must be merged so the tail query window is complete.
    tokens = tokens[:257]
    cache, _ = runtime.prefill(model, tokens, chunk_size=128, capture=True)
    assert all(layer.self_attn.last_queries.shape[-2] == 64 for layer in model.layers)
    scores = runtime.snapkv_scores(model, cache, window=window, kernel_size=5)
    # Independent float64 reference explicitly repeats KV heads and pools scores.
    for layer_index in (0, len(model.layers) - 1):
        layer = model.layers[layer_index]
        c = cache[layer_index]
        q = np.array(layer.self_attn.last_queries[:, :, -window:], dtype=np.float64)
        k = np.array(c.keys, dtype=np.float64)
        groups = q.shape[1] // k.shape[1]
        repeated = np.repeat(k, groups, axis=1)
        logits = (q @ repeated.swapaxes(-1, -2)) * layer.self_attn.attention.scale
        allowed = np.arange(len(tokens))[None] <= (
            len(tokens) - window + np.arange(window)
        )[:, None]
        logits = np.where(allowed, logits, -np.inf)
        mass = softmax(logits, axis=-1).reshape(
            1, k.shape[1], groups, window, len(tokens)
        ).mean(axis=(2, 3))[0, :, :-window]
        pooled = np.stack([np.convolve(row, np.ones(5) / 5, mode="same")
                           for row in mass])
        np.testing.assert_allclose(scores[layer_index][:, :-window], pooled,
                                   rtol=2e-4, atol=2e-7)
        assert np.all(scores[layer_index][:, -window:] > pooled.max())

    captured = [np.array(layer.self_attn.last_queries) for layer in model.layers]
    question_cache = runtime.fork_cache(cache)
    question = tokenizer.encode("What is the location of the bronze key?")
    model.model(mx.array(question)[None], cache=question_cache)
    mx.eval([c.state for c in question_cache])
    assert all(c.offset == len(tokens) for c in cache)
    for layer, old in zip(model.layers, captured):
        np.testing.assert_array_equal(np.array(layer.self_attn.last_queries), old)
    later_scores = runtime.snapkv_scores(model, cache, window=window, kernel_size=5)
    for old, new in zip(scores, later_scores):
        np.testing.assert_array_equal(old, new)


def test_pre_rope_inverse_matches_mlx_and_original_key_projection(runtime_model):
    mx, runtime, model, _, tokens = runtime_model
    import mlx.nn as nn
    from svkv.geometry import recover_pre_rope

    rng = np.random.default_rng(12)
    original = rng.normal(size=(2, 32768, 128)).astype(np.float32)
    rotated = np.array(nn.RoPE(128, traditional=False, base=1e6)(mx.array(original)))
    recovered = recover_pre_rope(rotated)
    error = float(np.max(np.abs(recovered - original)))
    print(f"pre_rope_fp32_32k_roundtrip_max_abs={error:.8g}")
    # MLX computes long-position angles in float32; the inverse uses float64
    # trigonometry. This permits their rounding difference at position 32767.
    np.testing.assert_allclose(recovered, original, rtol=0, atol=0.012)
    np.testing.assert_allclose(recovered[:, :128], original[:, :128], rtol=0, atol=5e-5)
    np.testing.assert_allclose(recover_pre_rope(rotated[:, -7:], positions=np.arange(32761, 32768)),
                               recovered[:, -7:], rtol=0, atol=0)

    # A real first-layer Qwen projection, before its native RoPE, is independently
    # accessible. Match the first 128-token prefill batch's quantized GEMM shape.
    cache, _ = runtime.prefill(model, tokens, chunk_size=128, capture=False)
    layer = model.layers[0]
    hidden = layer.input_layernorm(model.model.embed_tokens(mx.array(tokens[:128])[None]))
    attention = layer.self_attn.attention
    projection = attention.k_proj(hidden).reshape(1, 128, attention.n_kv_heads, -1)
    projection = np.array(projection.transpose(0, 2, 1, 3)[0], dtype=np.float32)
    recovered_projection = recover_pre_rope(np.array(cache[0].keys[0, :, :128]))
    error = float(np.max(np.abs(recovered_projection - projection)))
    print(f"pre_rope_real_fp16_projection_max_abs={error:.8g}")
    np.testing.assert_allclose(recovered_projection, projection, rtol=1e-3, atol=0.006)


def test_reserved_systems_cache_matches_quality_cache_logits(runtime_model):
    """Compare the serving backends at full budget and after headwise gathering."""
    mx, runtime, model, tokenizer, tokens = runtime_model
    from svkv.systems_cache import prefill_reserved, reserve_selected_cache, allocated_cache_bytes

    suffix = tokenizer.encode("Which cabinet contains the bronze key?")
    allowance = len(suffix) + 2
    quality, _ = runtime.prefill(model, tokens, chunk_size=128, capture=False)
    reserved, _ = prefill_reserved(model, tokens, chunk_size=128, capture=False,
                                   additional_tokens=allowance)
    assert all(c.capacity == len(tokens) + allowance for c in reserved)
    allocation = allocated_cache_bytes(reserved)
    for a, b in zip(quality, reserved):
        np.testing.assert_array_equal(np.array(a.keys), np.array(b.keys))
        np.testing.assert_array_equal(np.array(a.values), np.array(b.values))
    quality_logits = np.array(model(mx.array(suffix)[None], cache=quality).astype(mx.float32))
    reserved_logits = np.array(model(mx.array(suffix)[None], cache=reserved).astype(mx.float32))
    print(f"reserved_full_vs_quality_max_abs_logits={np.max(np.abs(quality_logits - reserved_logits)):.8g}")
    np.testing.assert_array_equal(reserved_logits, quality_logits)
    assert allocated_cache_bytes(reserved) == allocation

    indices = [np.stack([np.arange(h, 250 + h, 5)
                         for h in range(c.keys.shape[1])]) for c in quality]
    selected = runtime.fork_cache(quality, indices)
    selected_reserved = reserve_selected_cache(selected, additional_tokens=allowance)
    selected_allocation = allocated_cache_bytes(selected_reserved)
    selected_logits = np.array(model(mx.array(suffix)[None], cache=selected).astype(mx.float32))
    selected_reserved_logits = np.array(model(mx.array(suffix)[None], cache=selected_reserved).astype(mx.float32))
    print(f"reserved_selected_vs_quality_max_abs_logits={np.max(np.abs(selected_logits - selected_reserved_logits)):.8g}")
    np.testing.assert_array_equal(selected_reserved_logits, selected_logits)
    assert allocated_cache_bytes(selected_reserved) == selected_allocation
    assert all(c.offset == len(tokens) + 2 * len(suffix) for c in selected_reserved)
