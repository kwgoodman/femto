"Check that functions can handle list input"

import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal
import some_sums as ss

DTYPES = [np.float64, np.float32, np.int64, np.int32]


def test_list_input():
    "Check that functions can handle list input"
    for func in ss.get_functions('all'):
        if func.__name__ != 'replace':
            yield unit_maker, func


def lists(dtypes=DTYPES):
    "Iterator that yields lists to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1, 6), (2, 3)]}
    ss[3] = {'size':  6, 'shapes': [(1, 2, 3)]}
    ss[4] = {'size': 24, 'shapes': [(1, 2, 3, 4)]}
    for ndim in ss:
        size = ss[ndim]['size']
        shapes = ss[ndim]['shapes']
        a = np.arange(size)
        for shape in shapes:
            a = a.reshape(shape)
            for dtype in dtypes:
                yield a.astype(dtype).tolist()


def unit_maker(func):
    "Test that ss.xxx gives the same output as ss.slow.xxx for list input."
    msg = '\nfunc %s | input %s (%s) | shape %s\n'
    msg += '\nInput array:\n%s\n'
    name = func.__name__
    func0 = eval('ss.slow.%s' % name)
    for i, a in enumerate(lists()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                actual = func(a)
                desired = func0(a)
            except TypeError:
                actual = func(a, 2)
                desired = func0(a, 2)
        a = np.array(a)
        tup = (name, 'a'+str(i), str(a.dtype), str(a.shape), a)
        err_msg = msg % tup
        assert_array_almost_equal(actual, desired, err_msg=err_msg)
