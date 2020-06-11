import nose.tools
import numpy as np

import skeleton.io_tools as io_tools


def test_module_dir():
    d = io_tools.module_dir()
    assert d.endswith('skeleton'), d


def test_module_relative_path():
    nose.tools.assert_equals(
        io_tools.module_relative_path('io_tools_tests.py'),
        __file__)


def test_pad_int():
    np.testing.assert_array_equal(io_tools.padInt(5), "00000005")
