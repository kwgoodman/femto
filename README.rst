.. image:: https://travis-ci.org/kwgoodman/some_sums.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/some_sums
=========
some_sums
=========

What's the fastest way to sum a NumPy array?  I don't know---that's why I
created some_sums.

**some_sums**, written in C, contains several implementations of a sum
function. To keep things simple the input array must be at least 2d and the
axis of summation cannot be None.

My goal is to find fast ways to implement reduction functions (sum, mean,
std, max, nansum, etc.) that are bound by memory I/O. I chose summation as a
test case because very little time is spent with arithmetic which makes it
easier to measure improvements from things like manual loop unrolling,
software prefetching of data, and parallel processing.

some_sums is based on code from `bottleneck`_. It comes with a benchmark
suite::

    >>> ss.bench()
    some_sums performance benchmark
        some_sums 0.0.1dev; Numpy 1.11.2
        Speed is NumPy a.sum(axis) time divided by
            Some_sums sumXX(a, axis) time
        Score is len(speeds) / sum([1.0/s for s in speeds])

                    (1000,1000)(1000,1000)(1000,1000)(1000,1000)
                      float64     int64     float64     int64
                       axis=0     axis=0     axis=1     axis=1     score
        sum00          0.33       0.62       0.46       0.83       0.50
        sum01          0.46       0.69       1.13       1.37       0.76
        sum02          0.48       0.71       1.05       1.33       0.77
        sum03          0.43       0.59       1.33       1.63       0.74

Please help me avoid over optimizing for my particular operating system and
CPU. `Let me know`_ the benchmark results on your system.

If you have ideas on how to speed up the `code`_ then `share`_ them.

License
=======

some_sums is distributed under the GPL v3+. See the LICENSE file for details.

Requirements
============

- Python 2.7, 3.4, 3.5
- NumPy 1.11
- gcc
- nose

.. _bottleneck: https://github.com/kwgoodman/bottleneck
.. _code: https://github.com/kwgoodman/some_sums
.. _share: https://github.com/kwgoodman/some_sums/issues
.. _Let me know: https://github.com/kwgoodman/some_sums/issues