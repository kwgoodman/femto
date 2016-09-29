# flake8: noqa


# If you bork the build (e.g. by messing around with the templates),
# you still want to be able to import Bottleneck so that you can
# rebuild using the templates. So try to import the compiled Bottleneck
# functions to the top level, but move on if not successful.
try:
    from .reduce import (nansum, nanmean, nanstd, nanvar, nanmin, nanmax,
                         median, nanmedian, ss, nanargmin, nanargmax, anynan,
                         allnan)
except:
    pass

try:
    from . import slow
    from some_sums.version import __version__
    from some_sums.benchmark.bench import bench
    from some_sums.benchmark.bench_detailed import bench_detailed
    from some_sums.util import get_functions
except:
    pass

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No some_sums unit testing available.")
