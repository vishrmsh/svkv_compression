"""MLX Qwen2 physical cache compaction with absolute rotary positions.

Attention itself stays the installed mlx-lm implementation. This module wraps
it only to record the final context queries for a SnapKV-style baseline.
"""
from __future__ import annotations

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import ConcatenateKVCache


class CompactKVCache(ConcatenateKVCache):
    """Physical sequence length and next absolute RoPE position are distinct."""

    def update_and_fetch(self, keys, values):
        n_new = keys.shape[-2]
        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate((self.keys, keys), axis=-2)
            self.values = mx.concatenate((self.values, values), axis=-2)
        self.offset += n_new
        return self.keys, self.values

    def make_mask(self, n, return_array=False, window_size=None):
        if window_size is not None:
            raise NotImplementedError("Pilot supports full attention only")
        if n == 1:
            return None
        if not return_array:
            return "causal"
        physical = 0 if self.keys is None else self.keys.shape[-2]
        return mx.arange(physical + n)[None, :] <= (physical + mx.arange(n))[:, None]

    def size(self):
        return 0 if self.keys is None else self.keys.shape[-2]


class CaptureAttention(nn.Module):
    def __init__(self, attention, window=64):
        super().__init__()
        self.attention = attention
        self.window = window
        self.capture = False
        self.last_queries = None

    def __call__(self, x, mask=None, cache=None):
        if self.capture:
            start = max(0, x.shape[1] - self.window)
            q = self.attention.q_proj(x[:, start:])
            q = q.reshape(x.shape[0], x.shape[1] - start, self.attention.n_heads, -1).transpose(0, 2, 1, 3)
            self.last_queries = self.attention.rope(q, offset=cache.offset + start)
        return self.attention(x, mask, cache)


def install_capture(model):
    if model.model_type != "qwen2":
        raise ValueError("Validated runtime currently supports Qwen2/Qwen2.5 only")
    for layer in model.layers:
        layer.self_attn = CaptureAttention(layer.self_attn)


def prefill(model, tokens, chunk_size=512, capture=True):
    cache = [CompactKVCache() for _ in model.layers]
    t0 = time.perf_counter()
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        if 0 < len(tokens) - end < 64:
            end = len(tokens)
        # Make sure the final captured batch contains at least the observation window.
        for layer in model.layers:
            layer.self_attn.capture = capture and end == len(tokens)
        _ = model.model(mx.array(tokens[start:end])[None], cache=cache)
        mx.eval([c.state for c in cache])
        start = end
    for layer in model.layers:
        layer.self_attn.capture = False
    return cache, time.perf_counter() - t0


def snapkv_scores(model, cache, window=32, kernel_size=5):
    """Context-tail causal attention, GQA mean and average pooling (kvpress recipe).

    Uses the last min(window, final prefill chunk length) context queries. The
    runner ensures final chunks have >=window tokens. Questions are unseen.
    """
    from scipy.ndimage import uniform_filter1d
    scores = []
    for layer, c in zip(model.layers, cache):
        wrapper = layer.self_attn
        q = wrapper.last_queries[:, :, -window:]
        w = q.shape[-2]
        hkv = c.keys.shape[1]
        groups = q.shape[1] // hkv
        # GQA broadcasting without repeating the cached keys in memory.
        q = q.reshape(1, hkv, groups, w, -1)
        logits = (q.astype(mx.float32) @ c.keys[:, :, None].astype(mx.float32).swapaxes(-1, -2)) * wrapper.attention.scale
        n = c.keys.shape[-2]
        valid = mx.arange(n)[None, :] <= (n - w + mx.arange(w))[:, None]
        logits = mx.where(valid, logits, -mx.inf)
        weights = mx.softmax(logits, axis=-1)
        s = weights.mean(axis=(2, 3))[0, :, :n-w]
        s = np.array(s, dtype=np.float32)
        s = uniform_filter1d(s, size=kernel_size, axis=-1, mode="constant")
        s = np.pad(s, ((0, 0), (0, w)), constant_values=float(s.max() + 1))
        scores.append(s)
    return scores


def fork_cache(cache, indices=None):
    result = []
    for i, c in enumerate(cache):
        new = CompactKVCache()
        new.offset = c.offset
        if indices is None:
            new.keys, new.values = c.keys, c.values
        else:
            idx = mx.array(indices[i])[None, :, :, None]
            new.keys = mx.take_along_axis(c.keys, idx, axis=2)
            new.values = mx.take_along_axis(c.values, idx, axis=2)
        result.append(new)
    mx.eval([c.state for c in result])
    return result


def cache_bytes(cache):
    return sum(c.keys.nbytes + c.values.nbytes for c in cache)


def generate(model, tokenizer, suffix, cache, max_tokens=32):
    start = time.perf_counter()
    logits = model(mx.array(suffix)[None], cache=cache)[:, -1]
    token = int(mx.argmax(logits, axis=-1).item())
    ttft = time.perf_counter() - start
    output = []
    eos = set(tokenizer.eos_token_ids)
    decode_start = time.perf_counter()
    decode_steps = 0
    for step in range(max_tokens):
        if token in eos:
            break
        output.append(token)
        if step + 1 < max_tokens:
            logits = model(mx.array([[token]]), cache=cache)[:, -1]
            token = int(mx.argmax(logits, axis=-1).item())
            decode_steps += 1
    elapsed = time.perf_counter() - decode_start
    return tokenizer.decode(output), {
        "generated_tokens": len(output), "suffix_ttft_s": ttft,
        "decode_s": elapsed, "decode_forward_steps": decode_steps,
        "decode_tokens_per_s": decode_steps / elapsed if elapsed else 0,
        "generation_s": ttft + elapsed,
    }
