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

some_sums is based on code from `bottleneck`_. It comes with a benchmark
suite::

    >>> ss.bench()
    some_sums performance benchmark
        some_sums 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

            (1,1000) (1000,1000)(1000,1000)(1000,1000)(1000,1000)
            float64    float64     int64     float64     int64
             axis=1     axis=0     axis=0     axis=1     axis=1     score
    sum00     3.16       0.34       0.66       0.46       0.83       0.61
    sum01     3.12       0.35       0.66       0.47       1.06       0.64
    sum02     6.51       0.47       0.72       1.16       1.41       0.96
    sum03     6.53       0.60       0.92       1.11       1.37       1.10
    sum04     6.85       0.80       1.22       1.13       1.44       1.31
    sum05     9.65       1.20       1.23       1.42       1.40       1.58
    sum06    14.94       1.21       1.23       1.43       1.42       1.61
    p_sum01   1.30       1.46       1.86       2.86       3.63       1.91
    p_sum02   2.16       1.46       2.11       3.09       5.47       2.35
    p_sum03   1.46       2.38       3.07       3.82       5.33       2.66

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
        some_sums 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

             (1,1)     (10,10)    (40,40)    (60,60)   (100,100)
            float64    float64    float64    float64    float64
             axis=1     axis=1     axis=1     axis=1     axis=1     score
    sum00    15.10      11.25       2.41       1.39       0.94       2.13
    sum01    13.03      10.46       2.45       1.72       1.00       2.31
    sum02    12.82      12.10       5.35       3.74       2.43       4.87
    sum03    13.10      11.40       5.23       3.57       2.35       4.71
    sum04    14.94      13.49       5.70       3.81       2.43       5.05
    sum05    14.92      14.20       7.16       5.20       3.36       6.52
    sum06    14.78      10.97       7.56       5.03       2.99       6.07
    p_sum01   1.59       1.79       1.79       2.09       1.97       1.83
    p_sum02   1.42       1.71       1.81       2.22       2.74       1.88
    p_sum03   1.86       1.68       1.97       2.10       2.47       1.99

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