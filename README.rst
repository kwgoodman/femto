.. image:: https://travis-ci.org/kwgoodman/some_sums.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/some_sums
=========
some_sums
=========

What's the fastest way to sum a N-dimensional NumPy array along an axis?
I don't know---that's why I created *some_sums*.

`some_sums` is written in C and contains many versions of the same function.

Benchmark
=========

Only arrays with data type (dtype) int32, int64, float32, and float64 are
accelerated. All other dtypes result in calls to slower, unaccelerated
functions. In the rare case of a byte-swapped input array (e.g. a big-endian
array on a little-endian operating system) the function will not be
accelerated regardless of dtype.

Where
=====

===================   ========================================================
 code                 https://github.com/kwgoodman/some_sums
 mailing list         https://github.com/kwgoodman/some_sums/issues
===================   ========================================================

License
=======

`some_sums` is distributed under the GPL v3. See the LICENSE file for details.

Install
=======

Requirements:

======================== ====================================================
some_sums                Python 2.7, 3.4, 3.5; NumPy 1.11.0
Compile                  gcc, clang, MinGW or MSVC
Unit tests               nose
======================== ====================================================

To install `some_sums` on GNU/Linux, Mac OS X, et al.::

    $ sudo python setup.py install

To install `some_sums` on Windows, first install MinGW and add it to your
system path. Then install `some_sums` with the commands::

    python setup.py install --compiler=mingw32

Unit tests
==========

After you have installed `some_sums`, run the suite of unit tests::

    >>> import some_sums as ss
    >>> ss.test()
    <snip>
    Ran 7 tests in 0.335s
    OK
