
import numpy as np
import some_sums as ss
from .autotimeit import autotimeit

__all__ = ['bench']


def bench(dtype='float64', axes=[0, 1],
          shapes=[(1000, 1000), (1000, 1000)],
          nans=[False, False],
          order='C',
          functions=None):
    """
    Bottleneck benchmark.

    Parameters
    ----------
    dtype : str, optional
        Data type string such as 'float64', which is the default.
    axes : list, optional
        List of Axes along which to perform the calculations that are being
        benchmarked.
    shapes : list, optional
        A list of tuple shapes of input arrays to use in the benchmark.
    nans : list, optional
        A list of the bools (True or False), one for each tuple in the
        `shapes` list, that tells whether the input arrays should be randomly
        filled with one-third NaNs.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    functions : {list, None}, optional
        A list of strings specifying which functions to include in the
        benchmark. By default (None) all functions are included in the
        benchmark.

    Returns
    -------
    A benchmark report is printed to stdout.

    """

    if len(shapes) != len(nans):
        raise ValueError("`shapes` and `nans` must have the same length")
    if len(shapes) != len(axes):
        raise ValueError("`shapes` and `axes` must have the same length")

    dtype = str(dtype)

    tab = '    '

    # Header
    print('some_sums performance benchmark')
    print("%ssome_sums %s; Numpy %s" % (tab, ss.__version__, np.__version__))
    print("%sSpeed is a.sum(axis) time divided by ss.sumXX(a, axis) time"
          % tab)
    print("%sdtype = %s" % (tab, dtype))

    print('')
    header = [" "*14]
    for nan in nans:
        if nan:
            header.append("NaN".center(11))
        else:
            header.append("no NaN".center(11))
    print("".join(header))
    header = ["".join(str(shape).split(" ")).center(11) for shape in shapes]
    header = [" "*16] + header
    print("".join(header))
    header = ["".join(("axis=" + str(axis)).split(" ")).center(11)
              for axis in axes]
    header = [" "*16] + header
    print("".join(header))

    suite = benchsuite(shapes, dtype, axes, nans, order, functions)
    for test in suite:
        name = test["name"].ljust(12)
        fmt = tab + name + "%7.2f" + "%11.2f"*(len(shapes) - 1)
        speed = timer(test['statements'], test['setups'])
        print(fmt % tuple(speed))


def timer(statements, setups):
    speed = []
    if len(statements) != 2:
        raise ValueError("Two statements needed.")
    for setup in setups:
        with np.errstate(invalid='ignore'):
            t0 = autotimeit(statements[0], setup, repeat=4)
            t1 = autotimeit(statements[1], setup, repeat=4)
        speed.append(t1 / t0)
    return speed


def getarray(shape, dtype, nans=False, order='C'):
    a = np.arange(np.prod(shape), dtype=dtype)
    if nans and issubclass(a.dtype.type, np.inexact):
        a[::3] = np.nan
    else:
        rs = np.random.RandomState(shape)
        rs.shuffle(a)
    return np.array(a.reshape(*shape), order=order)


def benchsuite(shapes, dtype, axes, nans, order, functions):

    suite = []

    def getsetups(setup, shapes, nans, axes, order):
        template = """
        from some_sums.benchmark.bench import getarray
        a = getarray(%s, 'DTYPE', %s, '%s')
        axis=%s
        %s"""
        setups = []
        for shape, nan, axis in zip(shapes, nans, axes):
            setups.append(template % (str(shape), str(nan),
                          order, str(axis), setup))
        return setups

    # non-moving window functions
    funcs = ss.get_functions("reduce", as_string=True)
    for func in funcs:
        if functions is not None and func not in functions:
            continue
        run = {}
        run['name'] = func
        run['statements'] = ["func(a, axis)", "a.sum(axis)"]
        setup = "from some_sums import %s as func" % func
        run['setups'] = getsetups(setup, shapes, nans, axes, order)
        suite.append(run)

    # Strip leading spaces from setup code
    for i, run in enumerate(suite):
        for j in range(len(run['setups'])):
            t = run['setups'][j]
            t = '\n'.join([z.strip() for z in t.split('\n')])
            suite[i]['setups'][j] = t

    # Set dtype in setups
    for i, run in enumerate(suite):
        for j in range(len(run['setups'])):
            t = run['setups'][j]
            t = t.replace('DTYPE', dtype)
            suite[i]['setups'][j] = t

    # Set dtype in statements
    for i, run in enumerate(suite):
        for j in range(2):
            t = run['statements'][j]
            t = t.replace('DTYPE', dtype)
            suite[i]['statements'][j] = t

    return suite
