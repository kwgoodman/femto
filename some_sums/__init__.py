# flake8: noqa


# If you bork the build (e.g. by messing around with the templates),
# you still want to be able to import some_sums so that you can
# rebuild using the templates. So try to import the compiled some_sums
# functions to the top level, but move on if not successful.
try:
    from .sums import (sum00, sum01, sum02, sum03, sum04, sum05, sum06,
                       sum07, sum08)
except:
    pass

try:
    from some_sums.version import __version__
    from some_sums.benchmark import bench, bench_overhead, bench_3d
    from some_sums.util import get_functions
except:
    pass

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No some_sums unit testing available.")
