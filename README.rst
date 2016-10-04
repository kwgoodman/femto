.. image:: https://travis-ci.org/kwgoodman/some_sums.svg?branch=master
    :target: https://travis-ci.org/kwgoodman/some_sums
=========
some_sums
=========

What's the fastest way to sum a N-dimensional NumPy array along an axis?
I don't know---that's why I created some_sums.

**some_sums**, written in C, contains several implementations of a `sum`
function. To keep things simple the input array must have ndim > 1 and axis
cannot be None.

Benchmark
=========

Some_sums comes with a benchmark suite::

    >>> ss.bench()
    Some_sums performance benchmark
        some_sums 0.0.1dev; Numpy 1.11.0
        Speed is a.sum(axis) time divided by ss.sumXX(a, axis) time
        dtype = float64

                     no NaN     no NaN
                    (1000,1000)(1000,1000)
                       axis=0     axis=1
        sum00          0.33       0.46
        sum01          0.48       1.15
        sum02          0.47       1.07

Where
=====

===================   ========================================================
 code                 https://github.com/kwgoodman/some_sums
 mailing list         https://github.com/kwgoodman/some_sums/issues
===================   ========================================================

License
=======

some_sums is distributed under the GPL v3. See the LICENSE file for details.

Install
=======

Requirements:

======================== ====================================================
some_sums                Python 2.7, 3.4, 3.5; NumPy 1.11.0
Compile                  gcc, clang, MinGW or MSVC
Unit tests               nose
======================== ====================================================

To install some_sums on GNU/Linux, Mac OS X, et al.::

    $ sudo python setup.py install

To install some_sums on Windows, first install MinGW and add it to your
system path. Then install some_sums with the commands::

    python setup.py install --compiler=mingw32

Unit tests
==========

After you have installed some_sums, run the suite of unit tests::

    >>> import some_sums as ss
    >>> ss.test()
    <snip>
    Ran 9 tests in 0.635s
    OK