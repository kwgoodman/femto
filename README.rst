.. image:: https://travis-ci.org/kwgoodman/some_sums.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/some_sums
=========
some_sums
=========

What's the fastest way to sum a NumPy array along an axis?  I don't
know---that's why I created some_sums.

**some_sums**, written in C, contains several implementations of a `sum`
function. To keep things simple the input array must have ndim > 1 and the
axis to sum over cannot be None.

some_sums is based on code from `bottleneck`_ and comes with a benchmark
suite::

    >>> ss.bench()
    some_sums performance benchmark
        some_sums 0.0.1dev; Numpy 1.11.2
        Speed is NumPy a.sum(axis) time divided by
        Some_sums sumXX(a, axis) time

                    (1000,1000)(1000,1000)(1000,1000)(1000,1000)
                      float64     int64     float64     int64
                       axis=0     axis=0     axis=1     axis=1
        sum00          0.35       0.61       0.46       0.84
        sum01          0.46       0.68       1.13       1.37
        sum02          0.46       0.74       1.05       1.30

If you have ideas on how to speed up the `code`_ then `share`_ them.

License
=======

some_sums is distributed under the GPL v3+. See the LICENSE file for details.

Requirements
============

- Python 2.7, 3.4, 3.5
- NumPy 1.11
- gcc or clang
- nose

.. _bottleneck: https://github.com/kwgoodman/bottleneck
.. _code: https://github.com/kwgoodman/some_sums
.. _share: https://github.com/kwgoodman/some_sums/issues