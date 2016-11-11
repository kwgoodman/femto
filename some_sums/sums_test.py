"Test sums functions."

from itertools import permutations

import numpy as np
from numpy.testing import assert_array_almost_equal

import some_sums as ss

DTYPES = [np.float64, np.float32, np.int64, np.int32]


def test_sums():
    "test sums functions"
    for func in ss.get_functions():
        yield unit_maker, func, arrays


def unit_maker(func, arrays_func, decimal=5):
    "Test that ss.sumXX gives the same output as np.sum."
    fmt = '\nfunc %s | input %s (%s) | shape %s | axis %s | order %s\n'
    fmt += '\nInput array:\n%s\n'
    name = func.__name__
    func0 = np.sum
    for i, a in enumerate(arrays_func()):
        axes = range(-1, a.ndim)
        for axis in axes:
            # do not use a.copy() here because it will C order the array
            actual = func(a, axis=axis)
            desired = func0(a, axis=axis)
            tup = (name, 'a'+str(i), str(a.dtype), str(a.shape),
                   str(axis), array_order(a), a)
            err_msg = fmt % tup
            assert_array_almost_equal(actual, desired, decimal, err_msg)


def arrays():
    for a in array_iter():
        if a.ndim < 2:
            yield a
        elif a.ndim == 3:
            for axes in permutations(range(a.ndim)):
                yield np.transpose(a, axes)
        else:
            yield a
            yield a.T


def get_array_number(number):
    for i, a in enumerate(arrays()):
        if i == number:
            return a
    raise ValueError("Could not find array number")


def array_iter(dtypes=DTYPES):
    "Iterator that yields arrays to use for unit testing."

    # nan and inf
    nan = np.nan
    inf = np.inf

    yield np.ones((2, 0))
    yield np.ones((0, 2))
    yield np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float16)

    # automate a bunch of arrays to test
    shapes = [(2, 6), (3, 4), (2, 3, 4), (1, 2, 3, 4)]
    for seed in (1, 2):
        rs = np.random.RandomState(seed)
        for shape in shapes:
            for dtype in dtypes:
                a = np.arange(np.prod(shape), dtype=dtype)
                if issubclass(a.dtype.type, np.inexact):
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = inf
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = nan
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] *= -1
                rs.shuffle(a)
                yield a.reshape(shape)

    # non-contiguous arrays
    for dtype in dtypes:
        a = np.arange(12).reshape(4, 3).astype(dtype)
        yield a[::2]
        yield a[:, ::2]
        yield a[::2][:, ::2]
    for dtype in dtypes:
        a = np.arange(60).reshape(3, 4, 5).astype(dtype)
        for start in range(2):
            for step in range(1, 2):
                yield a[start::step]
                yield a[:, start::step]
                yield a[:, :, start::step]
                yield a[start::step][::2]
                yield a[start::step][::2][:, ::2]

    # test loop unrolling
    for ndim in (2,):
        rs = np.random.RandomState(ndim)
        for length in range(25):
            for dtype in dtypes:
                if ndim == 2:
                    a = np.arange(length * 2, dtype=dtype)
                    rs.shuffle(a)
                    yield a.reshape(2, -1)
                else:
                    raise ValueError("`ndim` must be 2")

    # big array
    yield rs.rand(1000, 1000)
    yield rs.rand(1000, 1001)
    yield rs.rand(1001, 1001)


def array_order(a):
    f = a.flags
    string = []
    if f.c_contiguous:
        string.append("C")
    if f.f_contiguous:
        string.append("F")
    if len(string) == 0:
        string.append("N")
    return ",".join(string)
