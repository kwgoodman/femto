# flake8: noqa


# If you bork the build (e.g. by messing around with the templates),
# you still want to be able to import femto so that you can
# rebuild using the templates. So try to import the compiled femto
# functions to the top level, but move on if not successful.
try:
    from .sums import (sum00, sum01, p_sum01, sum02, p_sum02, sum03, p_sum03,
                       sum04, p_sum04, sum10, sum11, sum12)
except:
    pass

try:
    from femto.version import __version__
    from femto.benchmark import *
    from femto.util import get_functions
except:
    pass

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No femto unit testing available.")
