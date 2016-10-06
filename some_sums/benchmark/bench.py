
import numpy as np
import some_sums as ss
from .autotimeit import autotimeit

__all__ = ['bench']


def bench(shapes=[(1000, 1000), (1000, 1000), (1000, 1000), (1000, 1000)],
          dtypes=['float64', 'int64', 'float64', 'int64'],
          axes=[0, 0, 1, 1],
          order='C',
          functions=None):
    """
    Bottleneck benchmark.

    Parameters
    ----------
    shapes : list, optional
        A list of tuple shapes of input arrays to use in the benchmark.
    dtypes : list, optional
        A list of data type strings such as ['float64', 'int64'].
    axes : list, optional
        List of axes along which to perform the calculations that are being
        benchmarked.
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

    if len(shapes) != len(axes):
        raise ValueError("`shapes` and `axes` must have the same length")
    if len(dtypes) != len(axes):
        raise ValueError("`dtypes` and `axes` must have the same length")

    tab = '    '

    # header
    print('some_sums performance benchmark')
    print("%ssome_sums %s; Numpy %s" % (tab, ss.__version__, np.__version__))
    print("%sSpeed is NumPy a.sum(axis) time divided by\n"
          "%sSome_sums sumXX(a, axis) time" % (tab, tab))
    print('')
    header = [" "*14]
    header = ["".join(str(shape).split(" ")).center(11) for shape in shapes]
    header = [" "*16] + header
    print("".join(header))
    header = ["".join((str(dtype)).split(" ")).center(11)
              for dtype in dtypes]
    header = [" "*16] + header
    print("".join(header))
    header = ["".join(("axis=" + str(axis)).split(" ")).center(11)
              for axis in axes]
    header = [" "*16] + header
    print("".join(header))

    suite = benchsuite(shapes, dtypes, axes, order, functions)
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
            t0 = autotimeit(statements[0], setup, repeat=3)
            t1 = autotimeit(statements[1], setup, repeat=3)
        speed.append(t1 / t0)
    return speed


def getarray(shape, dtype, order='C'):
    a = np.arange(np.prod(shape), dtype=dtype)
    rs = np.random.RandomState(shape)
    rs.shuffle(a)
    return np.array(a.reshape(*shape), order=order)


def benchsuite(shapes, dtypes, axes, order, functions):

    suite = []

    def getsetups(setup, shapes, dtypes, axes, order):
        template = """
        from some_sums.benchmark.bench import getarray
        a = getarray(%s, '%s', '%s')
        axis=%s
        %s"""
        setups = []
        for shape, dtype, axis in zip(shapes, dtypes, axes):
            s = template % (str(shape), dtype, order, str(axis), setup)
            s = '\n'.join([line.strip() for line in s.split('\n')])
            setups.append(s)
        return setups

    # add functions to suite
    funcs = ss.get_functions("reduce", as_string=True)
    for func in funcs:
        if functions is not None and func not in functions:
            continue
        run = {}
        run['name'] = func
        run['statements'] = ["func(a, axis)", "a.sum(axis)"]
        setup = "from some_sums import %s as func" % func
        run['setups'] = getsetups(setup, shapes, dtypes, axes, order)
        suite.append(run)

    return suite
