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
sum03, sum04), SSE3 (sum05), AVX (sum06), and OpenMP (p_sum01, p_sum02,
p_sum03).

some_sums, based on code from `bottleneck`_, comes with several benchmark
suites::

    >>> ss.bench_axis0()
    some_sums performance benchmark
        some_sums 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

          (1000,1000)(1000,1000)(1000,1000)(1000,1000)
            float64    float32     int64      int32
             axis=0     axis=0     axis=0     axis=0     score
    sum00     0.34       0.22       0.57       1.19       0.40
    sum01     0.35       0.22       0.64       1.19       0.40
    sum02     0.47       0.27       0.69       1.20       0.49
    sum03     0.61       0.41       0.90       1.88       0.70
    sum04     0.80       0.45       1.21       1.89       0.83
    sum05     1.20       0.45       1.21       1.89       0.91
    sum06     1.21       0.45       1.21       1.90       0.91
    p_sum01   1.41       0.21       1.97       1.29       0.58
    p_sum02   1.33       0.21       1.97       1.29       0.59
    p_sum03   1.45       0.62       2.65       3.71       1.36

    >>> ss.bench_axis1()
    some_sums performance benchmark
        some_sums 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

          (1000,1000)(1000,1000)(1000,1000)(1000,1000)
            float64    float32     int64      int32
             axis=1     axis=1     axis=1     axis=1     score
    sum00     0.47       0.44       0.82       1.90       0.65
    sum01     0.47       0.44       1.05       2.62       0.70
    sum02     1.12       1.29       1.37       4.40       1.52
    sum03     1.11       1.29       1.39       4.41       1.53
    sum04     1.10       1.29       1.35       4.36       1.50
    sum05     1.33       1.28       1.35       4.36       1.60
    sum06     1.38       1.28       1.35       4.36       1.61
    p_sum01   2.62       1.68       3.18       6.94       2.79
    p_sum02   3.32       4.09       4.56      13.74       4.78
    p_sum03   3.08       2.67       4.80      10.55       3.99

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

    >>> ss.bench_overhead_axis0()
    some_sums performance benchmark
        some_sums 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

            (10,10)    (10,10)    (10,10)    (10,10)
            float64    float32     int64      int32
             axis=0     axis=0     axis=0     axis=0     score
    sum00    13.88      12.96      13.34      16.08      13.97
    sum01    12.61      13.02      13.33      14.81      13.39
    sum02    15.46      14.63      14.46      15.95      15.10
    sum03    15.72      14.81      14.59      16.15      15.29
    sum04    15.60      14.19      14.87      16.17      15.17
    sum05    16.73      14.31      14.93      16.09      15.45
    sum06    17.03      14.42      14.98      16.27      15.61
    p_sum01   1.35       1.21       1.47       1.73       1.42
    p_sum02   1.44       0.90       1.59       1.85       1.35
    p_sum03   1.71       1.12       1.40       1.78       1.45

    >>> ss.bench_overhead_axis1()
    some_sums performance benchmark
        some_sums 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

            (10,10)    (10,10)    (10,10)    (10,10)
            float64    float32     int64      int32
             axis=1     axis=1     axis=1     axis=1     score
    sum00    13.57      12.63      12.78      15.65      13.56
    sum01    12.17      12.58      12.55      14.07      12.80
    sum02    14.67      13.66      13.37      14.84      14.11
    sum03    13.33      12.71      12.73      13.55      13.07
    sum04    16.38      15.70      15.28      16.45      15.94
    sum05    17.03      15.76      15.30      16.55      16.13
    sum06    12.64      15.42      15.12      16.59      14.79
    p_sum01   1.95       1.43       1.67       1.77       1.68
    p_sum02   1.52       1.63       1.47       1.74       1.58
    p_sum03   1.50       1.48       1.65       1.86       1.61

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