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

Some_sums comes with a benchmark suite::

    >>> ss.bench()
    Some_sums performance benchmark
        some_sums 0.0.1dev; Numpy 1.11.0
        Speed is a.sum(axis) time divided by ss.sumXX(a, axis) time
        dtype = float64

                     no NaN     no NaN     no NaN
                       (100,)  (1000,1000)(1000,1000)
                       axis=0     axis=0     axis=1
        sum00          18.5        0.1        0.5
        sum01          18.8        0.1        0.5

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