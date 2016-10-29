/*
    This file is part of some_sums.

    some_sums is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    some_sums is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with some_sums.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "sums.h"
#include "iterators.h"
#include <x86intrin.h>

/* sum00 ----------------------------------------------------------------- */

/* simple for loop */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
REDUCE(sum00, DTYPE0)
{
    npy_DTYPE0 asum;
    INIT(DTYPE0, DTYPE0)
    WHILE {
        asum = 0;
        FOR asum += AI(DTYPE0);
        YPP = asum;
        NEXT
    }
    return y;
}
/* dtype end */

REDUCE_MAIN(sum00)


/* sum01 ----------------------------------------------------------------- */

/* simple for loop with manual loop unrolling (X4) */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
REDUCE(sum01, DTYPE0)
{
    npy_DTYPE0 asum;
    INIT(DTYPE0, DTYPE0)
    if (LENGTH < 4) {
        WHILE {
            asum = 0;
            FOR asum += AI(DTYPE0);
            YPP = asum;
            NEXT
        }
    }
    else {
        WHILE {
            Py_ssize_t repeat = LENGTH - LENGTH % 4;
            npy_DTYPE0 s[4];
            s[0] = AX(DTYPE0, 0);
            s[1] = AX(DTYPE0, 1);
            s[2] = AX(DTYPE0, 2);
            s[3] = AX(DTYPE0, 3);
            Py_ssize_t i = 4;
            for (; i < repeat; i += 4) {
                s[0] += AX(DTYPE0, i);
                s[1] += AX(DTYPE0, i + 1);
                s[2] += AX(DTYPE0, i + 2);
                s[3] += AX(DTYPE0, i + 3);
            }
            for (; i < LENGTH; i++) {
                s[0] += AX(DTYPE0, i);
            }
            YPP = s[0] + s[1] + s[2] + s[3];
            NEXT
        }
    }
    return y;
}
/* dtype end */

REDUCE_MAIN(sum01)


/* sum02 ----------------------------------------------------------------- */

/* add special casing for summing along non-fast axis */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
static PyObject *
sum02_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        npy_DTYPE0 asum;
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                asum = 0;
                FOR asum += AI(DTYPE0);
                YPP = asum;
                NEXT
            }
        }
        else {
            WHILE {
                Py_ssize_t repeat = LENGTH - LENGTH % 4;
                npy_DTYPE0 s[4];
                s[0] = AX(DTYPE0, 0);
                s[1] = AX(DTYPE0, 1);
                s[2] = AX(DTYPE0, 2);
                s[3] = AX(DTYPE0, 3);
                Py_ssize_t i = 4;
                for (; i < repeat; i += 4) {
                    s[0] += AX(DTYPE0, i);
                    s[1] += AX(DTYPE0, i + 1);
                    s[2] += AX(DTYPE0, i + 2);
                    s[3] += AX(DTYPE0, i + 3);
                }
                for (; i < LENGTH; i++) {
                    s[0] += AX(DTYPE0, i);
                }
                YPP = s[0] + s[1] + s[2] + s[3];
                NEXT
            }
        }
    }
    else {
        INIT2(DTYPE0, DTYPE0)
        WHILE {
            FOR {
                YI(DTYPE0) += AI(DTYPE0);
            }
            NEXT2
        }
    }
    return y;
}
/* dtype end */

static PyObject *
sum02(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer02(args,
                     kwds,
                     sum02_float64,
                     sum02_float32,
                     sum02_int64,
                     sum02_int32);
}


/* sum03 ----------------------------------------------------------------- */

/* loop unrolling of special casing for summing along non-fast axis */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
static PyObject *
sum03_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        npy_DTYPE0 asum;
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                asum = 0;
                FOR asum += AI(DTYPE0);
                YPP = asum;
                NEXT
            }
        }
        else {
            WHILE {
                Py_ssize_t repeat = LENGTH - LENGTH % 4;
                npy_DTYPE0 s[4];
                s[0] = AX(DTYPE0, 0);
                s[1] = AX(DTYPE0, 1);
                s[2] = AX(DTYPE0, 2);
                s[3] = AX(DTYPE0, 3);
                Py_ssize_t i = 4;
                for (; i < repeat; i += 4) {
                    s[0] += AX(DTYPE0, i);
                    s[1] += AX(DTYPE0, i + 1);
                    s[2] += AX(DTYPE0, i + 2);
                    s[3] += AX(DTYPE0, i + 3);
                }
                for (; i < LENGTH; i++) {
                    s[0] += AX(DTYPE0, i);
                }
                YPP = s[0] + s[1] + s[2] + s[3];
                NEXT
            }
        }
    }
    else {
        INIT2(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                FOR {
                    YI(DTYPE0) += AI(DTYPE0);
                }
                NEXT2
            }
        }
        else {
            Py_ssize_t repeat = LENGTH - LENGTH % 4;
            WHILE {
                npy_intp i = 0;
                for (; i < repeat; i += 4) {
                    YX(DTYPE0, i) += AX(DTYPE0, i);
                    YX(DTYPE0, i + 1) += AX(DTYPE0, i + 1);
                    YX(DTYPE0, i + 2) += AX(DTYPE0, i + 2);
                    YX(DTYPE0, i + 3) += AX(DTYPE0, i + 3);
                }
                for (; i < LENGTH; i++) {
                    YX(DTYPE0, i) += AX(DTYPE0, i);
                }
                NEXT2
            }
        }
    }
    return y;
}
/* dtype end */

static PyObject *
sum03(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer02(args,
                     kwds,
                     sum03_float64,
                     sum03_float32,
                     sum03_int64,
                     sum03_int32);
}


/* sum04 ----------------------------------------------------------------- */

/* simd: sse3 */

/* copied from numpy; modified; do not use if LENGTH < peel possible */
static BN_INLINE npy_uintp
calc_peel(const void * addr, const npy_uintp esize, const npy_uintp alignment)
{
    const npy_uintp offset = (npy_uintp)addr & (alignment - 1);
    npy_uintp peel = offset ? (alignment - offset) / esize : 0;
    return peel;
}

/* dtype = [['float64']] */
static PyObject *
sum04_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        npy_DTYPE0 asum;
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 9 || !IS_CONTIGUOUS(a)) {
            /* could loop unroll here */
            WHILE {
                asum = 0;
                FOR asum += AI(DTYPE0);
                YPP = asum;
                NEXT
            }
        }
        else {
            const Py_ssize_t i_simd = LENGTH - LENGTH % 8;
            WHILE {
                double sum_simd, sum = 0.0;
                double *ad = (double *)it.pa;
                __m128d vsum0, vsum1, vsum2, vsum3;
                npy_uintp peel = calc_peel(ad, sizeof(double), 16);
                npy_intp i = 0;
                for (; i < peel; i++) {
                    sum += ad[i];
                }
                vsum0 = _mm_load_pd(&ad[peel + 0]);
                vsum1 = _mm_load_pd(&ad[peel + 2]);
                vsum2 = _mm_load_pd(&ad[peel + 4]);
                vsum3 = _mm_load_pd(&ad[peel + 6]);
                for (i = i + 8; i < i_simd + peel; i += 8)
                {
                    __m128d v0 = _mm_load_pd(&ad[i]);
                    __m128d v1 = _mm_load_pd(&ad[i + 2]);
                    __m128d v2 = _mm_load_pd(&ad[i + 4]);
                    __m128d v3 = _mm_load_pd(&ad[i + 6]);
                    vsum0 = _mm_add_pd(vsum0, v0);
                    vsum1 = _mm_add_pd(vsum1, v1);
                    vsum2 = _mm_add_pd(vsum2, v2);
                    vsum3 = _mm_add_pd(vsum3, v3);
                }
                vsum0 = _mm_add_pd(vsum0, vsum1);
                vsum1 = _mm_add_pd(vsum2, vsum3);
                vsum0 = _mm_add_pd(vsum0, vsum1);
                vsum0 = _mm_hadd_pd(vsum0, vsum0);
                _mm_storeh_pd(&sum_simd, vsum0);
                for (; i < LENGTH; i++) {
                    sum += ad[i];
                }
                YPP = sum + sum_simd;
                NEXT
            }
        }
    }
    else {
        INIT2(DTYPE0, DTYPE0)
        if (LENGTH < 9) {
            WHILE {
                FOR {
                    YI(DTYPE0) += AI(DTYPE0);
                }
                NEXT2
            }
        }
        else {
            if (!IS_CONTIGUOUS(a) || LENGTH & 1 || (npy_uintp)it.pa & 15) {
                const Py_ssize_t repeat = LENGTH - LENGTH % 4;
                WHILE {
                    npy_intp i = 0;
                    for (; i < repeat; i += 4) {
                        YX(DTYPE0, i) += AX(DTYPE0, i);
                        YX(DTYPE0, i + 1) += AX(DTYPE0, i + 1);
                        YX(DTYPE0, i + 2) += AX(DTYPE0, i + 2);
                        YX(DTYPE0, i + 3) += AX(DTYPE0, i + 3);
                    }
                    for (; i < LENGTH; i++) {
                        YX(DTYPE0, i) += AX(DTYPE0, i);
                    }
                    NEXT2
                }
            }
            else {
                const Py_ssize_t i_simd = LENGTH - LENGTH % 8;
                WHILE {
                    double *ad = (double *)it.pa;
                    double *yd = (double *)it.py;
                    npy_intp i = 0;
                    for (; i < i_simd; i += 8)
                    {
                        __m128d a0 = _mm_load_pd(&ad[i]);
                        __m128d a1 = _mm_load_pd(&ad[i + 2]);
                        __m128d a2 = _mm_load_pd(&ad[i + 4]);
                        __m128d a3 = _mm_load_pd(&ad[i + 6]);

                        __m128d y0 = _mm_load_pd(&yd[i]);
                        __m128d y1 = _mm_load_pd(&yd[i + 2]);
                        __m128d y2 = _mm_load_pd(&yd[i + 4]);
                        __m128d y3 = _mm_load_pd(&yd[i + 6]);

                        _mm_store_pd(&yd[i],     _mm_add_pd(y0, a0));
                        _mm_store_pd(&yd[i + 2], _mm_add_pd(y1, a1));
                        _mm_store_pd(&yd[i + 4], _mm_add_pd(y2, a2));
                        _mm_store_pd(&yd[i + 6], _mm_add_pd(y3, a3));
                    }
                    for (; i < LENGTH; i++) {
                        yd[i] += ad[i];
                    }
                    NEXT2
                }
            }
        }
    }
    return y;
}
/* dtype end */

/* dtype = [['float32'], ['int64'], ['int32']] */
static PyObject *
sum04_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        npy_DTYPE0 asum;
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                asum = 0;
                FOR asum += AI(DTYPE0);
                YPP = asum;
                NEXT
            }
        }
        else {
            WHILE {
                Py_ssize_t i;
                Py_ssize_t repeat = LENGTH - LENGTH % 4;
                npy_DTYPE0 s[4];
                s[0] = AX(DTYPE0, 0);
                s[1] = AX(DTYPE0, 1);
                s[2] = AX(DTYPE0, 2);
                s[3] = AX(DTYPE0, 3);
                for (i = 4; i < repeat; i += 4) {
                    s[0] += AX(DTYPE0, i);
                    s[1] += AX(DTYPE0, i + 1);
                    s[2] += AX(DTYPE0, i + 2);
                    s[3] += AX(DTYPE0, i + 3);
                }
                for (i = i; i < LENGTH; i++) {
                    s[0] += AX(DTYPE0, i);
                }
                YPP = s[0] + s[1] + s[2] + s[3];
                NEXT
            }
        }
    }
    else {
        INIT2(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                FOR {
                    YI(DTYPE0) += AI(DTYPE0);
                }
                NEXT2
            }
        }
        else {
            npy_intp i;
            Py_ssize_t repeat = LENGTH - LENGTH % 4;
            WHILE {
                for (i = 0; i < repeat; i += 4) {
                    YX(DTYPE0, i) += AX(DTYPE0, i);
                    YX(DTYPE0, i + 1) += AX(DTYPE0, i + 1);
                    YX(DTYPE0, i + 2) += AX(DTYPE0, i + 2);
                    YX(DTYPE0, i + 3) += AX(DTYPE0, i + 3);
                }
                for (i = i; i < LENGTH; i++) {
                    YX(DTYPE0, i) += AX(DTYPE0, i);
                }
                NEXT2
            }
        }
    }
    return y;
}
/* dtype end */

static PyObject *
sum04(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer02(args,
                     kwds,
                     sum04_float64,
                     sum04_float32,
                     sum04_int64,
                     sum04_int32);
}


/* sum05 ----------------------------------------------------------------- */

/* simd: avx */

/* dtype = [['float64']] */
static PyObject *
sum05_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        npy_DTYPE0 asum;
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 19 || !IS_CONTIGUOUS(a)) {
            /* could loop unroll here */
            WHILE {
                asum = 0;
                FOR asum += AI(DTYPE0);
                YPP = asum;
                NEXT
            }
        }
        else {
            Py_ssize_t i_simd = LENGTH - LENGTH % 16;
            WHILE {
                double sum_simd, sum = 0.0;
                double *ad = (double *)it.pa;
                __m256d vsum0, vsum1, vsum2, vsum3;
                npy_uintp peel = calc_peel(ad, sizeof(double), 32);
                npy_intp i = 0;
                for (; i < peel; i++) {
                    sum += ad[i];
                }
                vsum0 = _mm256_load_pd(&ad[peel + 0]);
                vsum1 = _mm256_load_pd(&ad[peel + 4]);
                vsum2 = _mm256_load_pd(&ad[peel + 8]);
                vsum3 = _mm256_load_pd(&ad[peel + 12]);
                for (i = i + 16; i < i_simd + peel; i += 16)
                {
                    __m256d v0 = _mm256_load_pd(&ad[i]);
                    __m256d v1 = _mm256_load_pd(&ad[i + 4]);
                    __m256d v2 = _mm256_load_pd(&ad[i + 8]);
                    __m256d v3 = _mm256_load_pd(&ad[i + 12]);
                    vsum0 = _mm256_add_pd(vsum0, v0);
                    vsum1 = _mm256_add_pd(vsum1, v1);
                    vsum2 = _mm256_add_pd(vsum2, v2);
                    vsum3 = _mm256_add_pd(vsum3, v3);
                }
                vsum0 = _mm256_add_pd(vsum0, vsum1);
                vsum1 = _mm256_add_pd(vsum2, vsum3);
                vsum0 = _mm256_add_pd(vsum0, vsum1);
	            vsum0 = _mm256_add_pd(vsum0,
                        _mm256_permute2f128_pd(vsum0, vsum0, 0x1));
	            _mm_store_sd(&sum_simd,
                             _mm_hadd_pd(_mm256_castpd256_pd128(vsum0),
                                         _mm256_castpd256_pd128(vsum0)));
                for (; i < LENGTH; i++) {
                    sum += ad[i];
                }
                YPP = sum + sum_simd;
                NEXT
            }
        }
    }
    else {
        INIT2(DTYPE0, DTYPE0)
        if (LENGTH < 9) {
            WHILE {
                FOR {
                    YI(DTYPE0) += AI(DTYPE0);
                }
                NEXT2
            }
        }
        else {
            if (!IS_CONTIGUOUS(a) || LENGTH & 1 || (npy_uintp)it.pa & 15) {
                const Py_ssize_t repeat = LENGTH - LENGTH % 4;
                WHILE {
                    npy_intp i = 0;
                    for (; i < repeat; i += 4) {
                        YX(DTYPE0, i) += AX(DTYPE0, i);
                        YX(DTYPE0, i + 1) += AX(DTYPE0, i + 1);
                        YX(DTYPE0, i + 2) += AX(DTYPE0, i + 2);
                        YX(DTYPE0, i + 3) += AX(DTYPE0, i + 3);
                    }
                    for (; i < LENGTH; i++) {
                        YX(DTYPE0, i) += AX(DTYPE0, i);
                    }
                    NEXT2
                }
            }
            else {
                const Py_ssize_t i_simd = LENGTH - LENGTH % 8;
                WHILE {
                    double *ad = (double *)it.pa;
                    double *yd = (double *)it.py;
                    npy_intp i = 0;
                    for (; i < i_simd; i += 8)
                    {
                        __m128d a0 = _mm_load_pd(&ad[i]);
                        __m128d a1 = _mm_load_pd(&ad[i + 2]);
                        __m128d a2 = _mm_load_pd(&ad[i + 4]);
                        __m128d a3 = _mm_load_pd(&ad[i + 6]);

                        __m128d y0 = _mm_load_pd(&yd[i]);
                        __m128d y1 = _mm_load_pd(&yd[i + 2]);
                        __m128d y2 = _mm_load_pd(&yd[i + 4]);
                        __m128d y3 = _mm_load_pd(&yd[i + 6]);

                        _mm_store_pd(&yd[i],     _mm_add_pd(y0, a0));
                        _mm_store_pd(&yd[i + 2], _mm_add_pd(y1, a1));
                        _mm_store_pd(&yd[i + 4], _mm_add_pd(y2, a2));
                        _mm_store_pd(&yd[i + 6], _mm_add_pd(y3, a3));
                    }
                    for (; i < LENGTH; i++) {
                        yd[i] += ad[i];
                    }
                    NEXT2
                }
            }
        }
    }
    return y;
}
/* dtype end */

/* dtype = [['float32'], ['int64'], ['int32']] */
static PyObject *
sum05_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        npy_DTYPE0 asum;
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                asum = 0;
                FOR asum += AI(DTYPE0);
                YPP = asum;
                NEXT
            }
        }
        else {
            WHILE {
                Py_ssize_t i;
                Py_ssize_t repeat = LENGTH - LENGTH % 4;
                npy_DTYPE0 s[4];
                s[0] = AX(DTYPE0, 0);
                s[1] = AX(DTYPE0, 1);
                s[2] = AX(DTYPE0, 2);
                s[3] = AX(DTYPE0, 3);
                for (i = 4; i < repeat; i += 4) {
                    s[0] += AX(DTYPE0, i);
                    s[1] += AX(DTYPE0, i + 1);
                    s[2] += AX(DTYPE0, i + 2);
                    s[3] += AX(DTYPE0, i + 3);
                }
                for (i = i; i < LENGTH; i++) {
                    s[0] += AX(DTYPE0, i);
                }
                YPP = s[0] + s[1] + s[2] + s[3];
                NEXT
            }
        }
    }
    else {
        INIT2(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                FOR {
                    YI(DTYPE0) += AI(DTYPE0);
                }
                NEXT2
            }
        }
        else {
            npy_intp i;
            Py_ssize_t repeat = LENGTH - LENGTH % 4;
            WHILE {
                for (i = 0; i < repeat; i += 4) {
                    YX(DTYPE0, i) += AX(DTYPE0, i);
                    YX(DTYPE0, i + 1) += AX(DTYPE0, i + 1);
                    YX(DTYPE0, i + 2) += AX(DTYPE0, i + 2);
                    YX(DTYPE0, i + 3) += AX(DTYPE0, i + 3);
                }
                for (i = i; i < LENGTH; i++) {
                    YX(DTYPE0, i) += AX(DTYPE0, i);
                }
                NEXT2
            }
        }
    }
    return y;
}
/* dtype end */

static PyObject *
sum05(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer02(args,
                     kwds,
                     sum05_float64,
                     sum05_float32,
                     sum05_int64,
                     sum05_int32);
}

/* sum06 ----------------------------------------------------------------- */

/* OpenMP */

static BN_INLINE char**
slice_starts(npy_intp *yshape, npy_intp *nits, PyArrayObject *a, int axis)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *strides = PyArray_STRIDES(a);

    npy_intp its;
    char *pa = PyArray_BYTES(a);
    npy_intp indices[NPY_MAXDIMS];
    npy_intp astrides[NPY_MAXDIMS];
    char **ppa;

    *nits = 1;
    for (i = 0; i < ndim; i++) {
        if (i == axis) {
            continue;
        }
        else {
            indices[j] = 0;
            astrides[j] = strides[i];
            yshape[j] = shape[i];
            *nits *= shape[i];
            j++;
        }
    }

    ppa = malloc(*nits * sizeof(char*));
    for (its = 0; its < *nits; its++) {
        ppa[its] = pa;
        for (i = ndim - 2; i > -1; i--) {
            if (indices[i] < yshape[i] - 1) {
                pa += astrides[i];
                indices[i]++;
                break;
            }
            pa -= indices[i] * astrides[i];
            indices[i] = 0;
        }
    }
    return ppa;
}

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
REDUCE(sum06, DTYPE0)
{
    int ndim = PyArray_NDIM(a);
    Py_ssize_t length = PyArray_DIM(a, axis);
    Py_ssize_t astride = PyArray_STRIDE(a, axis);
    npy_intp its;
    npy_intp nits;
    npy_intp yshape[NPY_MAXDIMS];
    char **ppa = slice_starts(yshape, &nits, a, axis);
    PyObject *y = PyArray_EMPTY(ndim - 1, yshape, NPY_DTYPE0, 0);
    npy_DTYPE0 *py = (npy_DTYPE0 *)PyArray_DATA((PyArrayObject *)y);
    #pragma omp parallel for private(its)
    for (its = 0; its < nits; its++) {
        npy_intp i;
        npy_DTYPE0 s = 0;
        for (i = 0; i < length; i++) {
            s += *(npy_DTYPE0 *)(ppa[its] + i * astride);
        }
        py[its] = s;
    }
    free(ppa);
    return y;
}
/* dtype end */

REDUCE_MAIN(sum06)


/* python strings -------------------------------------------------------- */

PyObject *pystr_a = NULL;
PyObject *pystr_axis = NULL;

static int
intern_strings(void) {
    pystr_a = PyString_InternFromString("a");
    pystr_axis = PyString_InternFromString("axis");
    return pystr_a && pystr_axis;
}

/* reducer --------------------------------------------------------------- */

static BN_INLINE int
parse_args(PyObject *args,
           PyObject *kwds,
           PyObject **a,
           PyObject **axis)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    const Py_ssize_t nkwds = kwds == NULL ? 0 : PyDict_Size(kwds);
    if (nkwds) {
        int nkwds_found = 0;
        PyObject *tmp;
        switch (nargs) {
            case 1: *a = PyTuple_GET_ITEM(args, 0);
            case 0: break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
        switch (nargs) {
            case 0:
                *a = PyDict_GetItem(kwds, pystr_a);
                if (*a == NULL) {
                    TYPE_ERR("Cannot find `a` keyword input");
                    return 0;
                }
                nkwds_found += 1;
            case 1:
                tmp = PyDict_GetItem(kwds, pystr_axis);
                if (tmp != NULL) {
                    *axis = tmp;
                    nkwds_found++;
                }
                break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
        if (nkwds_found != nkwds) {
            TYPE_ERR("wrong number of keyword arguments");
            return 0;
        }
        if (nargs + nkwds_found > 2) {
            TYPE_ERR("too many arguments");
            return 0;
        }
    }
    else {
        switch (nargs) {
            case 2:
                *axis = PyTuple_GET_ITEM(args, 1);
            case 1:
                *a = PyTuple_GET_ITEM(args, 0);
                break;
            default:
                TYPE_ERR("wrong number of arguments");
                return 0;
        }
    }

    return 1;

}

static PyObject *
reducer(PyObject *args,
        PyObject *kwds,
        fone_t f_float64,
        fone_t f_float32,
        fone_t f_int64,
        fone_t f_int32)
{

    int ndim;
    int axis;
    int dtype;

    PyArrayObject *a;

    PyObject *a_obj = NULL;
    PyObject *axis_obj = NULL;

    if (!parse_args(args, kwds, &a_obj, &axis_obj)) return NULL;

    /* convert to array if necessary */
    if PyArray_Check(a_obj) {
        a = (PyArrayObject *)a_obj;
    } else {
        a = (PyArrayObject *)PyArray_FROM_O(a_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        VALUE_ERR("Byte-swapped arrays are not supported");
        return NULL;
    }

    /* does user want to reduce over all axes? */
    if (axis_obj == Py_None) {
        VALUE_ERR("`axis` cannot be None");
        return NULL;
    }
    else if (axis_obj == NULL) {
        ndim = PyArray_NDIM(a);
        if (ndim < 2) {
            VALUE_ERR("ndim must be > 1");
            return NULL;
        }
        axis = PyArray_NDIM(a) - 1;
    }
    else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer or None");
            return NULL;
        }
        ndim = PyArray_NDIM(a);
        if (axis < 0) {
            axis += ndim;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds", axis);
                return NULL;
            }
        }
        else if (axis >= ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            return NULL;
        }
        if (ndim == 1) {
            VALUE_ERR("ndim must be > 1");
            return NULL;
        }
    }

    dtype = PyArray_TYPE(a);

    /* we are reducing an array with ndim > 1 over a single axis */
    if (dtype == NPY_FLOAT64) {
        return f_float64(a, axis);
    }
    else if (dtype == NPY_FLOAT32) {
        return f_float32(a, axis);
    }
    else if (dtype == NPY_INT64) {
        return f_int64(a, axis);
    }
    else if (dtype == NPY_INT32) {
        return f_int32(a, axis);
    }
    else {
        return PyArray_Sum(a, axis, dtype, NULL);
    }

}

static PyObject *
reducer02(PyObject *args,
          PyObject *kwds,
          fnf_t f_float64,
          fnf_t f_float32,
          fnf_t f_int64,
          fnf_t f_int32)
{

    int ndim;
    int axis;
    int dtype;
    int fast_axis;

    PyArrayObject *a;

    PyObject *a_obj = NULL;
    PyObject *axis_obj = NULL;

    if (!parse_args(args, kwds, &a_obj, &axis_obj)) return NULL;

    /* convert to array if necessary */
    if PyArray_Check(a_obj) {
        a = (PyArrayObject *)a_obj;
    } else {
        a = (PyArrayObject *)PyArray_FROM_O(a_obj);
        if (a == NULL) {
            return NULL;
        }
    }

    /* check for byte swapped input array */
    if PyArray_ISBYTESWAPPED(a) {
        VALUE_ERR("Byte-swapped arrays are not supported");
        return NULL;
    }

    /* does user want to reduce over all axes? */
    ndim = PyArray_NDIM(a);
    if (axis_obj == Py_None) {
        VALUE_ERR("`axis` cannot be None");
        return NULL;
    }
    else if (axis_obj == NULL) {
        if (ndim < 2) {
            VALUE_ERR("ndim must be > 1");
            return NULL;
        }
        axis = PyArray_NDIM(a) - 1;
    }
    else {
        axis = PyArray_PyIntAsInt(axis_obj);
        if (error_converting(axis)) {
            TYPE_ERR("`axis` must be an integer or None");
            return NULL;
        }
        if (axis < 0) {
            axis += ndim;
            if (axis < 0) {
                PyErr_Format(PyExc_ValueError,
                             "axis(=%d) out of bounds", axis);
                return NULL;
            }
        }
        else if (axis >= ndim) {
            PyErr_Format(PyExc_ValueError, "axis(=%d) out of bounds", axis);
            return NULL;
        }
        if (ndim == 1) {
            VALUE_ERR("ndim must be > 1");
            return NULL;
        }
    }

    if (C_CONTIGUOUS(a)) {
        fast_axis = ndim - 1;
    }
    else if (F_CONTIGUOUS(a)) {
        fast_axis = 0;
    }
    else {
        int i;
        fast_axis = 0;
        npy_intp *strides = PyArray_STRIDES(a);
        npy_intp min_stride = strides[0];
        for (i = 1; i < ndim; i++) {
            if (strides[i] < min_stride) {
                min_stride = strides[i];
                fast_axis = i;
            }
        }
    }

    dtype = PyArray_TYPE(a);

    if (dtype == NPY_FLOAT64) {
        return f_float64(a, axis, fast_axis);
    }
    else if (dtype == NPY_FLOAT32) {
        return f_float32(a, axis, fast_axis);
    }
    else if (dtype == NPY_INT64) {
        return f_int64(a, axis, fast_axis);
    }
    else if (dtype == NPY_INT32) {
        return f_int32(a, axis, fast_axis);
    }
    else {
        return PyArray_Sum(a, axis, dtype, NULL);
    }

}

/* docstrings ------------------------------------------------------------- */

static char module_doc[] = "some_sums's some sums.";

static char sum_doc[] =
/* MULTILINE STRING BEGIN
sum(a, axis=-1)

Sum of array elements along given axis. a.dim must be greater than 1.

The data type (dtype) of the output is the same as the input. On 64-bit
operating systems, 32-bit input is NOT upcast to 64-bit accumulator and
return values.

Parameters
----------
a : array_like
    Array containing numbers whose sum is desired. If `a` is not an
    array, a conversion is attempted.
axis : int, optional
    Axis along which the sum is computed. The default (axis=-1) is to
    compute the sum along the last axis.

Returns
-------
y : ndarray
    An array with the same shape as `a`, with the specified axis removed.

Notes
-----
No error is raised on overflow.

If positive or negative infinity are present the result is positive or
negative infinity. But if both positive and negative infinity are present,
the result is Not A Number (NaN).

Examples
--------

>>> a = np.array([[1, 2], [3, 4]])
>>> ss.sum(a, axis=0)
array([ 4,  6])

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
sums_methods[] = {
    {"sum00", (PyCFunction)sum00, VARKEY, sum_doc},
    {"sum01", (PyCFunction)sum01, VARKEY, sum_doc},
    {"sum02", (PyCFunction)sum02, VARKEY, sum_doc},
    {"sum03", (PyCFunction)sum03, VARKEY, sum_doc},
    {"sum04", (PyCFunction)sum04, VARKEY, sum_doc},
    {"sum05", (PyCFunction)sum05, VARKEY, sum_doc},
    {"sum06", (PyCFunction)sum06, VARKEY, sum_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
sums_def = {
   PyModuleDef_HEAD_INIT,
   "sums",
   module_doc,
   -1,
   sums_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_sums(void)
#else
#define RETVAL
initsums(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&sums_def);
    #else
        PyObject *m = Py_InitModule3("sums", sums_methods, module_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
