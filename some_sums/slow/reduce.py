import numpy as np


def _sum(a, axis=None):
    "Slow nansum function used for unaccelerated dtype."
    a = np.asarray(a)
    y = np.sum(a, axis=axis)
    if y.dtype != a.dtype:
        y = y.astype(a.dtype)
    return y

sum00 = _sum
sum01 = _sum
sum02 = _sum
