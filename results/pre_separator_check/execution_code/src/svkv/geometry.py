"""Key geometry helpers for the unscaled, full-dimension Qwen2 RoPE pilot."""
from __future__ import annotations

import numpy as np


def recover_pre_rope(keys: np.ndarray, theta: float = 1e6,
                     positions: np.ndarray | None = None) -> np.ndarray:
    """Undo Qwen2's nontraditional RoPE for keys shaped ``(heads, n, dim)``.

    By default the cache is an uncompressed prefill at absolute positions
    ``0..n-1``. Supply original absolute positions for an already-selected cache.
    This does not support partial rotation, RoPE scaling, or traditional adjacent
    pairs. The inverse recovers key geometry to input rounding accuracy; it does
    not undo fp16 rounding that already occurred in the forward rotation.
    The serving cache must retain its original rotated keys unchanged.
    """
    array = np.asarray(keys)
    if array.ndim != 3 or array.shape[-1] == 0 or array.shape[-1] % 2:
        raise ValueError("keys must have shape (heads, n, positive even dim)")
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    n, dim = array.shape[-2:]
    if positions is None:
        positions = np.arange(n, dtype=np.float64)
    else:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (n,) or not np.isfinite(positions).all():
            raise ValueError("positions must be a finite length-n vector")
    half = dim // 2
    frequencies = np.exp(-np.log(theta) * np.arange(half, dtype=np.float64) / half)
    angles = positions[:, None] * frequencies[None, :]
    cosines, sines = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
    left = array[..., :half].astype(np.float32)
    right = array[..., half:].astype(np.float32)
    return np.concatenate((left * cosines + right * sines,
                           right * cosines - left * sines), axis=-1)
