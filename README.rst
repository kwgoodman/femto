.. image:: https://travis-ci.org/kwgoodman/femto.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/femto
=========
femto
=========

**update: after upgrading ipython timeit now reports that the fastest run
when using OpenMP is many times faster than the slowest run. So I no longer
believe the OpenMP results. I guess it was too good to believe that we could
see a speed up on such small arrays**

What's the fastest way to sum a NumPy array?  I don't know---that's why I
created femto.

**femto**, written in C, contains several implementations of a sum
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

femto, based on code from `bottleneck`_, comes with several benchmark
suites::

    >>> ss.bench_axis0()
    femto performance benchmark
        femto 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by femto time
        Score is harmonic mean of speeds

          (1000,1000)(1000,1000)(1000,1000)(1000,1000)
            float64    float32     int64      int32
             axis=0     axis=0     axis=0     axis=0     score
    sum00     0.34       0.22       0.63       1.19       0.40
    sum01     0.35       0.22       0.65       1.20       0.41
    sum02     0.47       0.27       0.69       1.19       0.49
    sum03     0.56       0.45       0.87       1.54       0.69
    sum04     0.81       0.45       0.56       1.55       0.68
    sum10     0.80       0.45       1.20       1.90       0.83
    sum11     1.19       1.34       1.20       1.88       1.35
    sum12     1.19       0.45       1.21       1.89       0.91
    p_sum01   1.30       0.22       2.02       1.36       0.62
    p_sum02   1.38       0.23       2.41       1.36       0.63
    p_sum03   1.90       1.13       2.73       4.28       1.99
    p_sum04   3.22       1.07       2.71       4.33       2.16

    >>> ss.bench_axis1()
    femto performance benchmark
        femto 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by femto time
        Score is harmonic mean of speeds

          (1000,1000)(1000,1000)(1000,1000)(1000,1000)
            float64    float32     int64      int32
             axis=1     axis=1     axis=1     axis=1     score
    sum00     0.48       0.44       0.84       1.32       0.63
    sum01     0.47       0.45       1.05       1.85       0.68
    sum02     1.13       1.29       1.38       3.07       1.47
    sum03     1.13       1.27       1.37       3.07       1.47
    sum04     1.12       1.29       1.36       3.11       1.47
    sum10     1.12       1.28       1.37       3.06       1.47
    sum11     1.42       3.04       1.38       3.07       1.92
    sum12     1.45       1.29       1.39       3.03       1.59
    p_sum01   3.11       3.04       4.16       6.80       3.85
    p_sum02   4.37       4.37       5.48      10.65       5.45
    p_sum03   4.20       4.35       5.52      10.48       5.37
    p_sum04   4.43       4.30       5.49      10.36       5.43

I chose numpy.sum as a benchmark because it is fast and convenient. It
should be possible to beat NumPy's performance. That's because femto has
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
femto, which is a speed of 2/2.5 = 0.8.

Let's look at function call overhead by benchmarking with small input arrays::

    >>> ss.bench_overhead()
    femto performance benchmark
        femto 0.0.1.dev0; Numpy 1.11.2
        Speed is NumPy time divided by femto time
        Score is harmonic mean of speeds

            (10,10)    (10,10)   (100,100)  (100,100)
            float64    float64    float64    float64
             axis=0     axis=1     axis=0     axis=1     score
    sum00    16.11      16.11       1.12       0.98       1.96
    sum01    13.82      13.22       1.17       1.03       2.03
    sum02    15.83      15.91       2.76       2.51       4.51
    sum03    16.94      13.38       2.81       2.40       4.42
    sum04    17.85      13.45       4.53       2.38       5.19
    sum10    16.67      18.16       2.01       2.52       3.96
    sum11    18.33      18.62       3.33       3.52       5.78
    sum12    18.29      16.14       3.27       3.01       5.30
    p_sum01   1.75       1.73       2.59       2.20       2.01
    p_sum02   1.80       1.77       2.83       2.58       2.15
    p_sum03   1.83       1.87       2.94       2.55       2.21
    p_sum04   1.90       1.85       3.30       2.46       2.25

Please help me avoid over optimizing for my particular operating system, CPU,
and compiler. `Let me know`_ the benchmark results on your system. If you have
ideas on how to speed up the `code`_ then `share`_ them.

License
=======

femto is distributed under the GPL v3+. See the LICENSE file for details.

Requirements
============

Currently femto only compiles on GNU/Linux.

- SSE3, AVX, x86intrin.h, OpenMP
- Python 2.7, 3.4, 3.5
- NumPy 1.11
- gcc
- nose

.. _bottleneck: https://github.com/kwgoodman/bottleneck
.. _code: https://github.com/kwgoodman/femto
.. _share: https://github.com/kwgoodman/femto/issues
.. _pairwise summation: https://en.wikipedia.org/wiki/Pairwise_summation
.. _Let me know: https://github.com/kwgoodman/femto/issues
