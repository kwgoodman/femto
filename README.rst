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
easier to measure improvements from things like manual loop unrolling, SIMD,
and OpenMP.

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
    sum00     0.34       0.22       0.63       1.19       0.40
    sum01     0.35       0.22       0.64       1.19       0.40
    sum02     0.48       0.27       0.74       1.20       0.50
    sum03     0.69       0.44       0.92       1.53       0.73
    sum04     0.86       0.44       0.71       1.53       0.72
    sum10     0.80       0.45       1.21       1.89       0.83
    sum11     1.21       1.32       1.21       1.89       1.36
    sum12     1.22       0.45       1.22       1.89       0.91
    p_sum01   1.43       0.21       2.31       1.30       0.60
    p_sum02   1.60       0.21       2.45       1.28       0.62
    p_sum03   1.95       1.09       2.74       4.26       1.97
    p_sum04   3.07       1.04       2.77       4.21       2.12

    >>> ss.bench_axis1()
    some_sums performance benchmark
        some_sums 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

          (1000,1000)(1000,1000)(1000,1000)(1000,1000)
            float64    float32     int64      int32
             axis=1     axis=1     axis=1     axis=1     score
    sum00     0.46       0.44       0.85       1.32       0.63
    sum01     0.46       0.44       1.06       1.85       0.68
    sum02     1.14       1.29       1.41       3.09       1.49
    sum03     1.13       1.28       1.37       3.08       1.47
    sum04     1.13       1.28       1.38       3.08       1.48
    sum10     1.11       1.29       1.35       3.07       1.46
    sum11     1.39       3.00       1.36       3.07       1.89
    sum12     1.36       1.28       1.34       3.07       1.55
    p_sum01   3.04       3.02       3.98       6.86       3.78
    p_sum02   4.00       4.46       5.63      10.79       5.37
    p_sum03   4.09       4.33       5.49      10.62       5.32
    p_sum04   4.13       4.21       5.59      10.38       5.30

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
    sum00    20.65      14.58       1.74       0.95       3.27       2.44
    sum01    16.61      12.01       1.76       1.01       3.16       2.48
    sum02    16.66      14.20       3.92       2.47       6.97       5.35
    sum03    16.36      12.64       3.58       2.35       6.93       5.05
    sum04    16.43      12.64       3.35       2.33       6.89       4.93
    sum10    20.21      16.37       4.08       2.50       7.52       5.63
    sum11    19.97      17.10       5.63       3.36      11.05       7.42
    sum12    20.07      14.57       5.06       2.94      14.35       6.89
    p_sum01   1.56       1.59       2.09       2.33       1.24       1.68
    p_sum02   1.49       1.70       2.25       2.54       1.68       1.86
    p_sum03   1.63       1.68       2.11       2.00       1.64       1.79
    p_sum04   1.60       1.66       1.86       2.19       1.67       1.77

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
