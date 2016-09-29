"Check that functions can handle scalar input"

from numpy.testing import assert_array_almost_equal
import some_sums as ss


def unit_maker(func, func0, args=tuple()):
    "Test that ss.xxx gives the same output as ss.slow.xxx for scalar input."
    msg = '\nfunc %s | input %s\n'
    a = -9
    argsi = [a] + list(args)
    actual = func(*argsi)
    desired = func0(*argsi)
    err_msg = msg % (func.__name__, a)
    assert_array_almost_equal(actual, desired, err_msg=err_msg)


def test_scalar_input():
    "Test scalar input"
    funcs = ss.get_functions('reduce')
    for func in funcs:
        yield unit_maker, func, eval('ss.slow.%s' % func.__name__)
