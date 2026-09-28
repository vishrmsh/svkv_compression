"""Reserved append buffers for isolated systems measurements.

The quality harness deliberately keeps its independent concatenate-based cache.
This backend reserves known capacity once and follows mlx-lm's indexed-update
cache pattern. There is no capacity growth or full-cache concatenation during
measured suffix/decode forwards; MLX kernels can still allocate temporaries.
"""
from __future__ import annotations

import time

import mlx.core as mx


class ReservedAppendCache:
    """Physical length, allocated capacity, and absolute RoPE offset are distinct.

    The first update allocates the fixed backing arrays using incoming key/value
    shapes and dtypes. ``keys``, ``values`` and ``state`` expose only live rows.
    This small serving-only API intentionally does not provide serialization,
    trimming, rolling windows, or capacity growth.
    """

    def __init__(self, capacity: int, absolute_offset: int = 0):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("capacity must be a nonnegative integer")
        if isinstance(absolute_offset, bool) or not isinstance(absolute_offset, int) or absolute_offset < 0:
            raise ValueError("absolute_offset must be a nonnegative integer")
        self.capacity = capacity
        self.offset = absolute_offset
        self._length = 0
        self._keys = None
        self._values = None

    @property
    def keys(self):
        return None if self._keys is None else self._keys[..., :self._length, :]

    @property
    def values(self):
        return None if self._values is None else self._values[..., :self._length, :]

    @property
    def state(self):
        return self.keys, self.values

    @property
    def backing_state(self):
        return self._keys, self._values

    @property
    def allocated_nbytes(self):
        return 0 if self._keys is None else self._keys.nbytes + self._values.nbytes

    @property
    def logical_nbytes(self):
        return 0 if self._keys is None else self.keys.nbytes + self.values.nbytes

    @property
    def nbytes(self):
        return self.allocated_nbytes

    def size(self):
        return self._length

    def empty(self):
        return self._length == 0

    def update_and_fetch(self, keys, values):
        if keys.ndim != 4 or values.ndim != 4 or keys.shape[:3] != values.shape[:3]:
            raise ValueError("K/V must have matching (batch, heads, tokens, dim) axes")
        n_new = keys.shape[-2]
        end = self._length + n_new
        if end > self.capacity:
            raise ValueError(f"Reserved KV capacity exceeded: {end} > {self.capacity}")
        if self._keys is None:
            key_shape = (*keys.shape[:2], self.capacity, keys.shape[-1])
            value_shape = (*values.shape[:2], self.capacity, values.shape[-1])
            self._keys = mx.zeros(key_shape, dtype=keys.dtype)
            self._values = mx.zeros(value_shape, dtype=values.dtype)
        elif (keys.shape[:2] != self._keys.shape[:2]
              or values.shape[:2] != self._values.shape[:2]
              or keys.shape[-1] != self._keys.shape[-1]
              or values.shape[-1] != self._values.shape[-1]
              or keys.dtype != self._keys.dtype or values.dtype != self._values.dtype):
            raise ValueError("K/V shape and dtype must match the reserved buffers")
        self._keys[..., self._length:end, :] = keys
        self._values[..., self._length:end, :] = values
        self._length = end
        self.offset += n_new
        return self.keys, self.values

    def make_mask(self, n, return_array=False, window_size=None):
        if window_size is not None:
            raise NotImplementedError("Systems pilot supports full attention only")
        if n == 1:
            return None
        if not return_array:
            return "causal"
        return mx.arange(self._length + n)[None, :] <= (
            self._length + mx.arange(n)
        )[:, None]


def reserve_selected_cache(cache, additional_tokens):
    """Materialize selected logical rows into independent reserved storage."""
    if not isinstance(additional_tokens, int) or additional_tokens < 0:
        raise ValueError("additional_tokens must be a nonnegative integer")
    reserved = []
    for source in cache:
        n = source.size()
        target = ReservedAppendCache(n + additional_tokens,
                                     absolute_offset=int(source.offset) - n)
        target.update_and_fetch(source.keys, source.values)
        # Finish the copy while source arrays are alive. Returned caches retain
        # only independent backing buffers, not lazy graphs into the original.
        mx.eval(target.backing_state)
        reserved.append(target)
    return reserved


def allocated_cache_bytes(cache):
    return sum(item.allocated_nbytes for item in cache)


def prefill_reserved(model, tokens, chunk_size=512, capture=True, additional_tokens=0):
    """Prefill all arms into capacity N + known suffix/decode allowance.

    Buffer allocation and population are part of the returned prefill time. The
    full arm directly serves this cache; selected arms create smaller reserves.
    """
    if not tokens or chunk_size < 1 or additional_tokens < 0:
        raise ValueError("nonempty tokens, positive chunk size and nonnegative allowance required")
    cache = [ReservedAppendCache(len(tokens) + additional_tokens) for _ in model.layers]
    capture_window = max(layer.self_attn.window for layer in model.layers)
    started = time.perf_counter()
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        if 0 < len(tokens) - end < capture_window:
            end = len(tokens)
        for layer in model.layers:
            layer.self_attn.capture = capture and end == len(tokens)
        model.model(mx.array(tokens[start:end])[None], cache=cache)
        mx.eval([item.backing_state for item in cache])
        start = end
    for layer in model.layers:
        layer.self_attn.capture = False
    return cache, time.perf_counter() - started
