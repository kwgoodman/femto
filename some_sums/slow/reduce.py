import numpy as np

__all__ = ['sum00']


def _sum(a, axis=None):
    "Slow nansum function used for unaccelerated dtype."
    a = np.asarray(a)
    y = np.nansum(a, axis=axis)
    if y.dtype != a.dtype:
        y = y.astype(a.dtype)
    return y

sum00 = _sum
