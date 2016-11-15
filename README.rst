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
    sum00     0.34       0.22       0.64       1.62       0.42
    sum01     0.36       0.22       0.66       1.61       0.42
    sum02     0.48       0.27       0.71       1.63       0.51
    sum03     0.86       0.47       0.91       2.15       0.82
    sum04     0.80       0.45       1.22       2.57       0.85
    sum05     1.21       1.31       1.21       2.58       1.43
    sum06     1.19       0.45       1.21       2.57       0.94
    p_sum01   1.28       0.21       2.30       1.00       0.58
    p_sum02   1.51       0.21       2.35       0.78       0.56
    p_sum03   2.26       1.06       2.51       6.06       2.05

    >>> ss.bench_axis1()
    some_sums performance benchmark
        some_sums 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

          (1000,1000)(1000,1000)(1000,1000)(1000,1000)
            float64    float32     int64      int32
             axis=1     axis=1     axis=1     axis=1     score
    sum00     0.46       0.44       0.84       1.87       0.65
    sum01     0.46       0.44       1.05       2.64       0.70
    sum02     1.13       1.29       1.40       4.39       1.54
    sum03     1.15       1.29       1.40       4.42       1.55
    sum04     1.13       1.28       1.39       4.36       1.53
    sum05     1.41       2.97       1.41       4.37       2.02
    sum06     1.44       1.28       1.40       4.37       1.65
    p_sum01   2.16       3.04       3.34       9.70       3.35
    p_sum02   4.13       4.42       5.84      15.44       5.68
    p_sum03   3.71       4.33       3.81      11.88       4.72

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

             (1,1)     (10,10)    (60,60)   (100,100)   (1,1000)
            float64    float64    float64    float64    float64
             axis=1     axis=1     axis=1     axis=1     axis=1     score
    sum00    20.81      14.82       1.76       0.96       3.39       2.48
    sum01    16.82      13.18       1.85       1.01       3.28       2.53
    sum02    16.72      14.59       3.90       2.47       7.26       5.40
    sum03    17.11      13.63       3.76       2.38       7.29       5.24
    sum04    20.39      16.25       4.04       2.56       7.83       5.70
    sum05    20.28      17.03       5.41       3.43      11.50       7.45
    sum06    20.00      12.75       4.93       2.98      14.89       6.81
    p_sum01   1.44       1.53       2.04       2.29       1.31       1.64
    p_sum02   1.49       1.86       2.25       2.72       1.54       1.87
    p_sum03   1.75       1.52       2.16       2.46       1.62       1.84

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