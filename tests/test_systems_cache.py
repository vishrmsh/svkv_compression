"""Tiny-array Metal checks; no model loading or serving benchmark."""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("SVKV_RUN_METAL_TESTS") != "1",
    reason="Opt in with SVKV_RUN_METAL_TESTS=1 on a Metal-enabled Mac",
)


def test_reserved_appends_preserve_absolute_offset_payload_and_capacity():
    import mlx.core as mx
    from svkv.systems_cache import ReservedAppendCache

    cache = ReservedAppendCache(capacity=9, absolute_offset=95)
    prefix = np.arange(40, dtype=np.float32).reshape(1, 2, 5, 4)
    cache.update_and_fetch(mx.array(prefix), mx.array(prefix + 100))
    mx.eval(cache.backing_state)
    assert cache.size() == 5 and cache.offset == 100 and cache.capacity == 9
    assert cache.keys.shape[-2] == 5
    allocation = cache.allocated_nbytes
    assert allocation == 2 * 1 * 2 * 9 * 4 * 4
    assert cache.logical_nbytes == 2 * prefix.nbytes
    expected_keys, expected_values = prefix, prefix + 100
    for count in (2, 1, 1):
        new_keys = np.full((1, 2, count, 4), cache.offset, dtype=np.float32)
        new_values = new_keys + 100
        got_keys, got_values = cache.update_and_fetch(mx.array(new_keys), mx.array(new_values))
        mx.eval(cache.backing_state)
        expected_keys = np.concatenate((expected_keys, new_keys), axis=2)
        expected_values = np.concatenate((expected_values, new_values), axis=2)
        np.testing.assert_array_equal(np.array(got_keys), expected_keys)
        np.testing.assert_array_equal(np.array(got_values), expected_values)
        assert cache.allocated_nbytes == allocation
        assert cache.backing_state[0].shape[-2] == 9
    assert cache.offset == 104 and cache.size() == 9
    with pytest.raises(ValueError, match="capacity exceeded"):
        cache.update_and_fetch(mx.zeros((1, 2, 1, 4)), mx.zeros((1, 2, 1, 4)))
    assert cache.offset == 104 and cache.size() == 9


def test_reserved_conversion_copies_rows_and_masks_physical_length():
    import mlx.core as mx
    from svkv.runtime import CompactKVCache
    from svkv.systems_cache import reserve_selected_cache, allocated_cache_bytes

    source = CompactKVCache()
    source.update_and_fetch(mx.arange(24).astype(mx.float32).reshape(1, 2, 3, 4),
                            mx.ones((1, 2, 3, 4)))
    source.offset = 103
    original = np.array(source.keys)
    target, = reserve_selected_cache([source], additional_tokens=5)
    assert target.offset == 103 and target.size() == 3 and target.capacity == 8
    assert allocated_cache_bytes([target]) == 2 * 1 * 2 * 8 * 4 * 4
    assert target.logical_nbytes == source.keys.nbytes + source.values.nbytes
    np.testing.assert_array_equal(np.array(target.keys), original)
    expected_mask = np.arange(5)[None, :] <= (3 + np.arange(2))[:, None]
    np.testing.assert_array_equal(np.array(target.make_mask(2, return_array=True)), expected_mask)
    assert target.make_mask(2) == "causal"
    assert target.make_mask(1) is None
    target.update_and_fetch(mx.zeros((1, 2, 2, 4)), mx.zeros((1, 2, 2, 4)))
    mx.eval(target.backing_state)
    np.testing.assert_array_equal(np.array(source.keys), original)
    assert source.offset == 103 and source.size() == 3
    assert target.offset == 105 and target.size() == 5
