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
    sum00     0.28       0.22       0.50       1.62       0.37
    sum01     0.29       0.22       0.48       1.63       0.37
    sum02     0.40       0.27       0.59       1.64       0.47
    sum03     0.67       0.46       0.83       2.15       0.75
    sum04     0.80       0.45       1.21       2.57       0.86
    sum05     1.17       0.45       1.21       2.57       0.94
    sum06     1.18       0.45       1.21       2.57       0.94
    p_sum01   0.41       0.21       0.60       0.97       0.41
    p_sum02   0.40       0.21       0.62       0.89       0.40
    p_sum03   1.63       0.87       1.87       5.27       1.61

    >>> ss.bench_axis1()
    some_sums performance benchmark
        some_sums 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by some_sums time
        Score is harmonic mean of speeds

          (1000,1000)(1000,1000)(1000,1000)(1000,1000)
            float64    float32     int64      int32
             axis=1     axis=1     axis=1     axis=1     score
    sum00     0.46       0.44       0.83       1.91       0.65
    sum01     0.47       0.44       1.06       2.63       0.70
    sum02     1.11       1.29       1.36       4.41       1.51
    sum03     1.12       1.29       1.36       4.44       1.53
    sum04     1.14       1.29       1.42       4.36       1.55
    sum05     1.40       1.28       1.41       4.38       1.64
    sum06     1.47       1.29       1.37       4.37       1.66
    p_sum01   3.02       3.05       3.71       9.47       3.87
    p_sum02   3.64       3.85       4.75      15.19       4.94
    p_sum03   2.77       4.21       5.09      14.00       4.61

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
    sum00    14.62      11.33       1.64       0.95       3.10       2.34
    sum01    12.97      10.14       1.77       1.02       3.01       2.43
    sum02    12.70      12.00       3.92       2.47       6.34       5.10
    sum03    12.55      11.16       3.58       2.33       6.30       4.82
    sum04    14.60      13.19       3.88       2.50       6.63       5.25
    sum05    14.54      13.50       5.06       3.30       9.24       6.65
    sum06    14.23      10.74       4.84       2.98      14.48       6.46
    p_sum01   1.38       1.50       2.05       2.17       1.19       1.57
    p_sum02   1.41       1.56       2.08       2.60       1.34       1.69
    p_sum03   1.62       1.40       2.08       2.39       1.42       1.70

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