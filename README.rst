.. image:: https://travis-ci.org/kwgoodman/some_sums.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/some_sums
=========
some_sums
=========

What's the fastest way to sum a NumPy array?  I don't know---that's why I
created some_sums.

**some_sums**, written in C, contains several implementations of a sum
function. To keep things simple the input array must be at least 2d and the
axis of summation cannot be None. Limiting ourselves to the 1d case would
have be even simpler. But I am interested in both summing along an axis
where the array elements are closely packed in memory (e.g. axis=-1 of a
C contiguous array) and where they are widely spaced (axis=0). Both cases
require different optimizations.

My goal is to find fast ways to implement reduction functions (sum, mean,
std, max, nansum, etc.) that are bound by memory I/O. I chose summation as a
test case because very little time is spent with arithmetic which makes it
easier to measure improvements from things like manual loop unrolling (sum02,
sum03, sum04), SSE3 (sum05), AVX (sum06), and OpenMP (p_sum00, p_sum01).

some_sums is based on code from `bottleneck`_. It comes with a benchmark
suite::

    >>> ss.bench()
    some_sums performance benchmark
        some_sums 0.0.1dev; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

            (1,1000) (1000,1000)(1000,1000)(1000,1000)(1000,1000)
            float64    float64     int64     float64     int64
             axis=1     axis=0     axis=0     axis=1     axis=1     score
    sum00     3.35       0.35       0.63       0.46       0.83       0.61
    sum01     3.26       0.35       0.65       0.46       1.04       0.64
    sum02     7.21       0.47       0.71       1.16       1.41       0.96
    sum03     7.68       0.64       0.91       1.16       1.39       1.14
    sum04     7.59       0.83       1.20       1.09       1.40       1.31
    sum05    11.48       1.22       1.22       1.40       1.42       1.59
    sum06    14.90       1.20       1.20       1.45       1.41       1.60
    p_sum01   1.41       1.06       2.10       2.79       3.63       1.81
    p_sum02   1.85       1.48       2.28       4.32       5.59       2.42

I chose numpy.sum as a benchmark because it is fast and convenient. It
should be possible to beat NumPy's performance. That's because some_sums has
an unfair advantage. We will not duplicate the `pairwise summation`_ NumPy
uses to deal with the accumulated round-off error in floating point arrays.

The overall fastest function is the one with the highest benchmark score.
Let's consider the case where we benchmark each function with two arrays
(five are used by default in the benchmark) and the speeds are 0.5 (half as
fast as NumPy) and 2.0 (twice as fast). What should the overall score be? Some
possibilities are the mean (1.25, which is faster than NumPy), the geometric
mean (1.0, same as NumPy), or the harmonic mean (0.8, slower). I chose the
harmonic mean. If a NumPy program spends equal time summing the two benchmark
arrays, each 1 unit of time, then it will take 1/2 + 2 units of time with
some_sums, which is a speed of 2/2.5 = 0.8.

Let's look at function call overhead by benchmarking with small input arrays::

    >>> ss.bench_overhead()
    some_sums performance benchmark
        some_sums 0.0.1dev; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

             (1,1)     (10,10)    (40,40)    (60,60)   (100,100)
            float64    float64    float64    float64    float64
             axis=1     axis=1     axis=1     axis=1     axis=1     score
    sum00    21.42      15.13       3.23       1.81       0.99       2.51
    sum01    17.70      14.99       3.27       1.97       1.05       2.64
    sum02    17.49      16.34       6.34       4.28       2.61       5.60
    sum03    20.94      16.33       6.17       4.13       2.54       5.51
    sum04    21.09      16.46       6.18       4.11       2.54       5.51
    sum05    20.80      17.40       8.29       5.50       3.43       7.15
    sum06    20.85      14.64       7.27       4.81       2.95       6.24
    p_sum01   1.83       1.37       2.19       2.53       2.79       2.01
    p_sum02   1.72       1.88       2.40       2.93       3.33       2.30

Please help me avoid over optimizing for my particular operating system, CPU,
and compiler. `Let me know`_ the benchmark results on your system. If you have
ideas on how to speed up the `code`_ then `share`_ them.

License
=======

some_sums is distributed under the GPL v3+. See the LICENSE file for details.

Requirements
============

Currently some_sums only compiles on GNU/Linux. `Please help`_ us with getting
it to compile on OSX and Windows.

- SSE3, AVX, x86intrin.h, OpenMP
- Python 2.7, 3.4, 3.5
- NumPy 1.11
- gcc
- nose

.. _bottleneck: https://github.com/kwgoodman/bottleneck
.. _code: https://github.com/kwgoodman/some_sums
.. _share: https://github.com/kwgoodman/some_sums/issues
.. _pairwise summation: https://en.wikipedia.org/wiki/Pairwise_summation
.. _Let me know: https://github.com/kwgoodman/some_sums/issues
.. _Please help: https://github.com/kwgoodman/some_sums/issues/1
