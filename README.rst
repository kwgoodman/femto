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
easier to measure improvements from things like manual loop unrolling (sum01,
sum02, sum03), SSE3 (sum04), AVX (sum05), and OpenMP (sum07, sum08).

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
    sum00     3.45       0.28       0.49       0.46       0.83       0.54
    sum01     7.48       0.37       0.50       1.09       1.28       0.76
    sum02     8.44       0.67       1.01       1.11       1.46       1.20
    sum03     8.33       0.84       1.24       1.08       1.31       1.31
    sum04    12.93       1.14       1.22       1.44       1.37       1.56
    sum05    15.03       1.24       1.15       1.36       1.33       1.55
    sum06     3.46       0.26       0.54       0.47       1.04       0.55
    sum07     1.43       1.34       2.11       2.92       4.28       2.00
    sum08     1.87       1.47       2.32       4.24       6.03       2.44

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
    sum00    22.80      14.22       2.54       1.66       0.99       2.35
    sum01    22.50      17.29       6.27       3.97       2.48       5.45
    sum02    22.21      17.09       6.41       4.07       2.56       5.58
    sum03    21.83      17.11       6.40       4.07       2.53       5.55
    sum04    22.17      18.43       8.45       5.67       3.59       7.44
    sum05    22.00      13.78       7.46       5.20       3.06       6.49
    sum06    18.67      15.32       3.30       1.89       1.05       2.62
    sum07     1.71       1.91       2.26       2.23       2.79       2.12
    sum08     1.90       1.91       2.48       2.06       3.51       2.25

Please help me avoid over optimizing for my particular operating system, CPU,
and compiler. `Let me know`_ the benchmark results on your system. If you have
ideas on how to speed up the `code`_ then `share`_ them.

License
=======

some_sums is distributed under the GPL v3+. See the LICENSE file for details.

Requirements
============

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
