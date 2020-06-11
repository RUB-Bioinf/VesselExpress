import nose.tools
import numpy as np

import skeleton.rotational_operators as ops

RAND_ARR = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=bool)


def test_rotate_3D_90():
    # test if rot_3D_90 raises assertion error correctly
    with nose.tools.assert_raises(AssertionError):
        ops.rot_3D_90(RAND_ARR[0:1])
    expected_sum = RAND_ARR.sum()
    obtained_sum = ops.rot_3D_90(RAND_ARR).sum()
    nose.tools.assert_equal(expected_sum, obtained_sum)


def test_rot_3D_90_identity():
    np.testing.assert_array_equal(RAND_ARR, RAND_ARR)

    for axis in ['x', 'y', 'z']:
        # FUll rotations should be identity.
        np.testing.assert_array_equal(
            RAND_ARR,
            # do 1 + 3 to avoid the %4 skip.
            ops.rot_3D_90(
                ops.rot_3D_90(RAND_ARR, rot_axis=axis, k=1),
                rot_axis=axis, k=3))

        np.testing.assert_array_equal(
            ops.rot_3D_90(RAND_ARR, rot_axis=axis, k=3),
            ops.rot_3D_90(
                ops.rot_3D_90(RAND_ARR, rot_axis=axis, k=1),
                rot_axis=axis, k=2))


def test_get_directions_list():
    # test if the border point after is rotated to the expected direction
    test_array = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 1, 0]]], dtype=np.uint8)
    directions_list = ops.get_directions_list(test_array)
    # expected index where 1 occurs after one of the rotation in 12 directions
    expected_results = [1, 23, 15, 5, 9, 25, 3, 19, 17, 21, 11, 7]

    for index, (expected_result, direction) in enumerate(zip(expected_results, directions_list)):
        nose.tools.assert_true(direction.reshape(27).tolist()[expected_result],
                               msg="fails at %i th direction" % index)
        nose.tools.assert_true(direction.sum(), msg="fails at %i th direction" % index)


def test_rotations_are_unique():
    test_array = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 1, 0]]], dtype=np.uint8)
    directions_list = ops.get_directions_list(test_array)

    for n, dir1 in enumerate(directions_list):
        for m, dir2 in enumerate(directions_list):
            if n == m:
                continue

            with nose.tools.assert_raises(AssertionError):
                np.testing.assert_array_equal(dir1, dir2)
            nose.tools.assert_equal(dir1.sum(), 1)
