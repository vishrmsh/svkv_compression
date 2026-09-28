import numpy as np
import pytest

from svkv.geometry import recover_pre_rope


def test_pre_rope_uses_absolute_positions_for_sparse_cache():
    # Nontraditional pairs are halves: (0,2), (1,3). With theta=1 all
    # frequencies are one; positions zero and pi/2 have a direct exact reference.
    original = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]], dtype=np.float32)
    rotated = np.array([[[1., 2., 3., 4.], [-7., -8., 5., 6.]]], dtype=np.float32)
    restored = recover_pre_rope(rotated, theta=1., positions=np.array([0., np.pi/2]))
    np.testing.assert_allclose(restored, original, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(rotated[0, 1], [-7., -8., 5., 6.])


def test_pre_rope_rejects_unsupported_shape_and_position_length():
    with pytest.raises(ValueError, match="positive even dim"):
        recover_pre_rope(np.zeros((2, 5, 3)))
    with pytest.raises(ValueError, match="length-n"):
        recover_pre_rope(np.zeros((2, 5, 4)), positions=np.arange(4))
    with pytest.raises(ValueError, match="theta"):
        recover_pre_rope(np.zeros((2, 5, 4)), theta=0)
