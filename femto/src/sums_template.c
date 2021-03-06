/*
    This file is part of femto.

    femto is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    femto is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with femto.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "sums.h"
#include "iterators.h"

/* sum00 ----------------------------------------------------------------- */

/* simple for loop in the style of bottleneck 1.2.0 */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
REDUCE(sum00, DTYPE0)
{
    INIT(DTYPE0, DTYPE0)
    WHILE {
        npy_DTYPE0 asum = 0;
        FOR asum += AI(DTYPE0);
        YPP = asum;
        NEXT
    }
    return y;
}
/* dtype end */

REDUCE_MAIN(sum00)

/* sum01, p_sum01 -------------------------------------------------------- */

/* It would be a lot of work to have a separate code base for single-threaded
 * and multi-threaded versions of the same function. Is there a way to use
 * the same code base for both versions? I think I came up with one that
 * doesn't sacrifice much performance.
 *
 * p_sum01 is the multi-threaded (OpenMP) version of sum00. And sum01 is the
 * single-threaded version of sum00. Note that sum01 and p_sum01 different by
 * only one line of code.*/

struct _piter {
    Py_ssize_t length;  /* a.shape[axis] */
    Py_ssize_t astride; /* a.strides[axis] */
    npy_intp   nits;    /* number of iterations iterator plans to make */
    npy_intp   yshape[NPY_MAXDIMS];    /* a.shape, a.shape[axis] removed */
    char       **ppa;    /* array of pointers to start of each slice */
};
typedef struct _piter piter;

static BN_INLINE void
init_piter(piter *it, PyArrayObject *a, int axis, PyObject **y, int ydtype)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    char *pa = PyArray_BYTES(a);
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *strides = PyArray_STRIDES(a);
    npy_intp yshape[NPY_MAXDIMS];
    npy_intp indices[NPY_MAXDIMS];
    it->length = shape[axis];
    it->astride = strides[axis];
    it->nits = 1;
    for (i = 0; i < ndim; i++) {
        indices[i] = 0;
        if (i != axis) {
            it->nits *= shape[i];
            yshape[j] = shape[i];
            j++;
        }
    }
    it->ppa = malloc(it->nits * sizeof(char*));
    for (j = 0; j < it->nits; j++) {
        it->ppa[j] = pa;
        for (i = ndim - 1; i > -1; i--) {
            if (i == axis) continue;
            if (indices[i] < shape[i] - 1) {
                pa += strides[i];
                indices[i]++;
                break;
            }
            pa -= indices[i] * strides[i];
            indices[i] = 0;
        }
    }
    *y = PyArray_EMPTY(ndim - 1, yshape, ydtype, 0);
}

#define P_INIT(dtype) \
    npy_intp its; \
    PyObject *y; \
    npy_##dtype *py; \
    piter it; \
    init_piter(&it, a, axis, &y, NPY_##dtype); \
    py = (npy_##dtype *)PyArray_DATA((PyArrayObject *)y);

#define P_RETURN \
    free(it.ppa); \
    return y;

#define A(dtype, i) \
    *(npy_##dtype *)(it.ppa[its] + (i) * it.astride)

/* repeat = {'NAME': ['sum01', 'p_sum01'],
             'PARALLEL': ['', '#pragma omp parallel for']} */
/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
REDUCE(NAME, DTYPE0)
{
    P_INIT(DTYPE0)
    PARALLEL
    for (its = 0; its < it.nits; its++) {
        npy_intp i;
        npy_DTYPE0 s = 0;
        for (i = 0; i < it.length; i++) {
            s += A(DTYPE0, i);
        }
        py[its] = s;
    }
    P_RETURN
}
/* dtype end */

REDUCE_MAIN(NAME)
/* repeat end */

/* sum02, p_sum02 -------------------------------------------------------- */

/* loop unrolling (x4) of sum01 and p_sum01 */

/* repeat = {'NAME': ['sum02', 'p_sum02'],
             'PARALLEL': ['', '#pragma omp parallel for']} */
/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
REDUCE(NAME, DTYPE0)
{
    P_INIT(DTYPE0)
    if (it.length < 4) {
        PARALLEL
        for (its = 0; its < it.nits; its++) {
            npy_intp i;
            npy_DTYPE0 s = 0;
            for (i = 0; i < it.length; i++) {
                s += A(DTYPE0, i);
            }
            py[its] = s;
        }
    }
    else {
        Py_ssize_t i_unroll = it.length - it.length % 4;
        PARALLEL
        for (its = 0; its < it.nits; its++) {
            Py_ssize_t i = 4;
            npy_DTYPE0 s[4];
            s[0] = A(DTYPE0, 0);
            s[1] = A(DTYPE0, 1);
            s[2] = A(DTYPE0, 2);
            s[3] = A(DTYPE0, 3);
            for (; i < i_unroll; i += 4) {
                s[0] += A(DTYPE0, i);
                s[1] += A(DTYPE0, i + 1);
                s[2] += A(DTYPE0, i + 2);
                s[3] += A(DTYPE0, i + 3);
            }
            for (; i < it.length; i++) {
                s[0] += A(DTYPE0, i);
            }
            py[its] = s[0] + s[1] + s[2] + s[3];
        }
    }
    P_RETURN
}
/* dtype end */

REDUCE_MAIN(NAME)
/* repeat end */

/* sum03, p_sum03 -------------------------------------------------------- */

/* add special casing for summing along non-fast axis */

#define N03 8

struct _piter2 {
    Py_ssize_t fast_length;
    npy_intp   fast_stride;
    npy_intp   fast_ystride;
    Py_ssize_t length;
    Py_ssize_t astride;
    npy_intp   nits4;
    npy_intp   nits;
    char       **ppa;
    char       **ppy;
};
typedef struct _piter2 piter2;

static BN_INLINE void
init_piter2(piter2 *it, PyArrayObject *a, int axis, PyObject **y, int ydtype,
            int fast_axis)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    char *pa = PyArray_BYTES(a);
    char *py;
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *strides = PyArray_STRIDES(a);
    const npy_intp *ystrides;
    npy_intp yshape[NPY_MAXDIMS];
    npy_intp astrides[NPY_MAXDIMS];
    npy_intp indices[NPY_MAXDIMS];
    npy_intp fast_nits;
    npy_intp fast_nits4;

    it->length = shape[axis];
    it->astride = strides[axis];
    it->fast_length = shape[fast_axis];
    it->fast_stride = strides[fast_axis];

    it->nits = 1;
    it->nits4 = 1;

    fast_nits4 = (it->fast_length - it->fast_length % N03) / N03;
    fast_nits = it->fast_length - N03 * fast_nits4;
    for (i = 0; i < ndim; i++) {
        indices[i] = 0;
        if (i != axis) {
            if (i == fast_axis) {
                it->nits4 *= fast_nits4;
                it->nits *= fast_nits;
            } else {
                it->nits4 *= shape[i];
                it->nits *= shape[i];
            }
            astrides[j] = strides[i];
            yshape[j] = shape[i];
            j++;
        }
    }
    it->nits += it->nits4;

    *y = PyArray_EMPTY(ndim - 1, yshape, ydtype, 0);
    py = PyArray_BYTES((PyArrayObject *)*y);
    ystrides = PyArray_STRIDES((PyArrayObject *)*y);

    it->ppa = malloc(2 * it->nits * sizeof(char*));
    it->ppy = &it->ppa[it->nits];

    fast_axis = fast_axis < axis ? fast_axis : fast_axis - 1;
    yshape[fast_axis] = N03 * fast_nits4;
    it->fast_ystride = ystrides[fast_axis];
    j = 0;
    for (; j < it->nits4; j++) {
        it->ppa[j] = pa;
        it->ppy[j] = py;
        for (i = ndim - 2; i > -1; i--) {
            if (i == fast_axis) {
                if (indices[i] < yshape[i] - N03) {
                    indices[i] += N03;
                    pa += N03 * astrides[i];
                    py += N03 * ystrides[i];
                    break;
                }
            }
            else {
                if (indices[i] < yshape[i] - 1) {
                    indices[i]++;
                    pa += astrides[i];
                    py += ystrides[i];
                    break;
                }
            }
            pa -= indices[i] * astrides[i];
            py -= indices[i] * ystrides[i];
            indices[i] = 0;
        }
    }
    yshape[fast_axis] = fast_nits;
    for (; j < it->nits; j++) {
        it->ppa[j] = pa + N03 * fast_nits4 * astrides[fast_axis];
        it->ppy[j] = py + N03 * fast_nits4 * ystrides[fast_axis];
        for (i = ndim - 2; i > -1; i--) {
            if (indices[i] < yshape[i] - 1) {
                indices[i]++;
                pa += astrides[i];
                py += ystrides[i];
                break;
            }
            pa -= indices[i] * astrides[i];
            py -= indices[i] * ystrides[i];
            indices[i] = 0;
        }
    }
}

#define P_INIT2(dtype) \
    npy_intp its; \
    PyObject *y; \
    piter2 it; \
    init_piter2(&it, a, axis, &y, NPY_##dtype, fast_axis); \

#define AP(dtype, p) \
    *(npy_##dtype *)(it.ppa[its] + i * it.astride + (p) * it.fast_stride)

#define YP(dtype, p) \
    *(npy_##dtype *)(it.ppy[its] + (p) * it.fast_ystride)

/* repeat = {'NAME': ['sum03', 'p_sum03'],
             'PARALLEL': ['', '#pragma omp parallel for']} */
/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
static PyObject *
NAME_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    if (axis == fast_axis) {
        P_INIT(DTYPE0)
        if (it.length < 4) {
            PARALLEL
            for (its = 0; its < it.nits; its++) {
                npy_intp i;
                npy_DTYPE0 s = 0;
                for (i = 0; i < it.length; i++) {
                    s += A(DTYPE0, i);
                }
                py[its] = s;
            }
        }
        else {
            Py_ssize_t i_unroll = it.length - it.length % 4;
            PARALLEL
            for (its = 0; its < it.nits; its++) {
                Py_ssize_t i = 4;
                npy_DTYPE0 s[4];
                s[0] = A(DTYPE0, 0);
                s[1] = A(DTYPE0, 1);
                s[2] = A(DTYPE0, 2);
                s[3] = A(DTYPE0, 3);
                for (; i < i_unroll; i += 4) {
                    s[0] += A(DTYPE0, i);
                    s[1] += A(DTYPE0, i + 1);
                    s[2] += A(DTYPE0, i + 2);
                    s[3] += A(DTYPE0, i + 3);
                }
                for (; i < it.length; i++) {
                    s[0] += A(DTYPE0, i);
                }
                py[its] = s[0] + s[1] + s[2] + s[3];
            }
        }
        return y;
    }
    else {
        P_INIT2(DTYPE0)
        PARALLEL
        for (its = 0; its < it.nits4; its++) {
            Py_ssize_t i = 0;
            npy_DTYPE0 s[N03];
            s[0] = AP(DTYPE0, 0);
            s[1] = AP(DTYPE0, 1);
            s[2] = AP(DTYPE0, 2);
            s[3] = AP(DTYPE0, 3);
            s[4] = AP(DTYPE0, 4);
            s[5] = AP(DTYPE0, 5);
            s[6] = AP(DTYPE0, 6);
            s[7] = AP(DTYPE0, 7);
            for (i = 1; i < it.length; i++) {
                s[0] += AP(DTYPE0, 0);
                s[1] += AP(DTYPE0, 1);
                s[2] += AP(DTYPE0, 2);
                s[3] += AP(DTYPE0, 3);
                s[4] += AP(DTYPE0, 4);
                s[5] += AP(DTYPE0, 5);
                s[6] += AP(DTYPE0, 6);
                s[7] += AP(DTYPE0, 7);
            }
            YP(DTYPE0, 0) = s[0];
            YP(DTYPE0, 1) = s[1];
            YP(DTYPE0, 2) = s[2];
            YP(DTYPE0, 3) = s[3];
            YP(DTYPE0, 4) = s[4];
            YP(DTYPE0, 5) = s[5];
            YP(DTYPE0, 6) = s[6];
            YP(DTYPE0, 7) = s[7];
        }
        for (its = it.nits4; its < it.nits; its++) {
            npy_intp i;
            npy_DTYPE0 s = 0;
            for (i = 0; i < it.length; i++) {
                s += AP(DTYPE0, 0);
            }
            YP(DTYPE0, 0) = s;
        }
        free(it.ppa);
        return y;
    }
}
/* dtype end */

static PyObject *
NAME(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer02(args,
                     kwds,
                     NAME_float64,
                     NAME_float32,
                     NAME_int64,
                     NAME_int32);
}
/* repeat end */

/* sum04, p_sum04 -------------------------------------------------------- */

/* add sse3 to sum03 */

/* repeat = {'NAME': ['sum04', 'p_sum04'],
             'PARALLEL': ['', '#pragma omp parallel for']} */
/* dtype = [['float64']] */
static PyObject *
NAME_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    if (axis == fast_axis) {
        P_INIT(DTYPE0)
        if (it.length < 4) {
            PARALLEL
            for (its = 0; its < it.nits; its++) {
                npy_intp i;
                npy_DTYPE0 s = 0;
                for (i = 0; i < it.length; i++) {
                    s += A(DTYPE0, i);
                }
                py[its] = s;
            }
        }
        else {
            Py_ssize_t i_unroll = it.length - it.length % 4;
            PARALLEL
            for (its = 0; its < it.nits; its++) {
                Py_ssize_t i = 4;
                npy_DTYPE0 s[4];
                s[0] = A(DTYPE0, 0);
                s[1] = A(DTYPE0, 1);
                s[2] = A(DTYPE0, 2);
                s[3] = A(DTYPE0, 3);
                for (; i < i_unroll; i += 4) {
                    s[0] += A(DTYPE0, i);
                    s[1] += A(DTYPE0, i + 1);
                    s[2] += A(DTYPE0, i + 2);
                    s[3] += A(DTYPE0, i + 3);
                }
                for (; i < it.length; i++) {
                    s[0] += A(DTYPE0, i);
                }
                py[its] = s[0] + s[1] + s[2] + s[3];
            }
        }
        return y;
    }
    else {
        npy_intp fast_length = PyArray_DIM(a, fast_axis);
        char *pa = PyArray_BYTES(a);
        PyObject *y;
        if (!(C_CONTIGUOUS(a) || PyArray_NDIM(a) == 2) || fast_length & 1 ||
            (npy_uintp)pa & 15) {
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
            return y;
        }
        else {
            P_INIT2(DTYPE0)
            npy_intp a_offset = it.astride / sizeof(double);
            PARALLEL
            for (its = 0; its < it.nits4; its++) {
                Py_ssize_t i;
                double *ad = (double *)it.ppa[its];
                double *yd = (double *)it.ppy[its];
                __m128d s[4];
                s[0] = _mm_load_pd(&ad[0]);
                s[1] = _mm_load_pd(&ad[2]);
                s[2] = _mm_load_pd(&ad[4]);
                s[3] = _mm_load_pd(&ad[6]);
                for (i = 1; i < it.length; i++) {
                    s[0] += _mm_load_pd(&ad[0 + i * a_offset]);
                    s[1] += _mm_load_pd(&ad[2 + i * a_offset]);
                    s[2] += _mm_load_pd(&ad[4 + i * a_offset]);
                    s[3] += _mm_load_pd(&ad[6 + i * a_offset]);
                }
                /* works for arrays that are 2d or C contiguous or both */
                _mm_store_pd(&yd[0], s[0]);
                _mm_store_pd(&yd[2], s[1]);
                _mm_store_pd(&yd[4], s[2]);
                _mm_store_pd(&yd[6], s[3]);
                /* works for all arrays but is slow:
                _mm_storel_pd(&yd[0 * it.fast_ystride/sizeof(double)], s[0]);
                _mm_storeh_pd(&yd[1 * it.fast_ystride/sizeof(double)], s[0]);
                _mm_storel_pd(&yd[2 * it.fast_ystride/sizeof(double)], s[1]);
                _mm_storeh_pd(&yd[3 * it.fast_ystride/sizeof(double)], s[1]);
                _mm_storel_pd(&yd[4 * it.fast_ystride/sizeof(double)], s[2]);
                _mm_storeh_pd(&yd[5 * it.fast_ystride/sizeof(double)], s[2]);
                _mm_storel_pd(&yd[6 * it.fast_ystride/sizeof(double)], s[3]);
                _mm_storeh_pd(&yd[7 * it.fast_ystride/sizeof(double)], s[3]);
                */
            }
            for (its = it.nits4; its < it.nits; its++) {
                npy_intp i;
                npy_DTYPE0 s = 0;
                for (i = 0; i < it.length; i++) {
                    s += AP(DTYPE0, 0);
                }
                YP(DTYPE0, 0) = s;
            }
            free(it.ppa);
            return y;
        }
    }
}
/* dtype end */
/* repeat end */

/* repeat = {'NAME': ['sum04', 'p_sum04'],
             'PARALLEL': ['', '#pragma omp parallel for']} */
/* dtype = [['float32'], ['int64'], ['int32']] */
static PyObject *
NAME_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    if (axis == fast_axis) {
        P_INIT(DTYPE0)
        if (it.length < 4) {
            PARALLEL
            for (its = 0; its < it.nits; its++) {
                npy_intp i;
                npy_DTYPE0 s = 0;
                for (i = 0; i < it.length; i++) {
                    s += A(DTYPE0, i);
                }
                py[its] = s;
            }
        }
        else {
            Py_ssize_t i_unroll = it.length - it.length % 4;
            PARALLEL
            for (its = 0; its < it.nits; its++) {
                Py_ssize_t i = 4;
                npy_DTYPE0 s[4];
                s[0] = A(DTYPE0, 0);
                s[1] = A(DTYPE0, 1);
                s[2] = A(DTYPE0, 2);
                s[3] = A(DTYPE0, 3);
                for (; i < i_unroll; i += 4) {
                    s[0] += A(DTYPE0, i);
                    s[1] += A(DTYPE0, i + 1);
                    s[2] += A(DTYPE0, i + 2);
                    s[3] += A(DTYPE0, i + 3);
                }
                for (; i < it.length; i++) {
                    s[0] += A(DTYPE0, i);
                }
                py[its] = s[0] + s[1] + s[2] + s[3];
            }
        }
        return y;
    }
    else {
        P_INIT2(DTYPE0)
        PARALLEL
        for (its = 0; its < it.nits4; its++) {
            Py_ssize_t i = 0;
            npy_DTYPE0 s[N03];
            s[0] = AP(DTYPE0, 0);
            s[1] = AP(DTYPE0, 1);
            s[2] = AP(DTYPE0, 2);
            s[3] = AP(DTYPE0, 3);
            s[4] = AP(DTYPE0, 4);
            s[5] = AP(DTYPE0, 5);
            s[6] = AP(DTYPE0, 6);
            s[7] = AP(DTYPE0, 7);
            for (i = 1; i < it.length; i++) {
                s[0] += AP(DTYPE0, 0);
                s[1] += AP(DTYPE0, 1);
                s[2] += AP(DTYPE0, 2);
                s[3] += AP(DTYPE0, 3);
                s[4] += AP(DTYPE0, 4);
                s[5] += AP(DTYPE0, 5);
                s[6] += AP(DTYPE0, 6);
                s[7] += AP(DTYPE0, 7);
            }
            YP(DTYPE0, 0) = s[0];
            YP(DTYPE0, 1) = s[1];
            YP(DTYPE0, 2) = s[2];
            YP(DTYPE0, 3) = s[3];
            YP(DTYPE0, 4) = s[4];
            YP(DTYPE0, 5) = s[5];
            YP(DTYPE0, 6) = s[6];
            YP(DTYPE0, 7) = s[7];
        }
        for (its = it.nits4; its < it.nits; its++) {
            npy_intp i;
            npy_DTYPE0 s = 0;
            for (i = 0; i < it.length; i++) {
                s += AP(DTYPE0, 0);
            }
            YP(DTYPE0, 0) = s;
        }
        free(it.ppa);
        return y;
    }
}
/* dtype end */

static PyObject *
NAME(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer02(args,
                     kwds,
                     NAME_float64,
                     NAME_float32,
                     NAME_int64,
                     NAME_int32);
}
/* repeat end */


/* sum10 ----------------------------------------------------------------- */

/* loop unrolling of special casing for summing along non-fast axis */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
static PyObject *
sum10_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                npy_DTYPE0 asum = 0;
                FOR asum += AI(DTYPE0);
                YPP = asum;
                NEXT
            }
        }
        else {
            WHILE {
                Py_ssize_t i = 4;
                Py_ssize_t repeat = LENGTH - LENGTH % 4;
                npy_DTYPE0 s[4];
                s[0] = AX(DTYPE0, 0);
                s[1] = AX(DTYPE0, 1);
                s[2] = AX(DTYPE0, 2);
                s[3] = AX(DTYPE0, 3);
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
sum10(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer02(args,
                     kwds,
                     sum10_float64,
                     sum10_float32,
                     sum10_int64,
                     sum10_int32);
}


/* sum11 ----------------------------------------------------------------- */

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
sum11_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 9 || !IS_CONTIGUOUS(a)) {
            /* could loop unroll here */
            WHILE {
                npy_DTYPE0 asum = 0;
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

/* dtype = [['float32']] */
static PyObject *
sum11_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 17 || !IS_CONTIGUOUS(a)) {
            /* could loop unroll here */
            WHILE {
                npy_DTYPE0 asum = 0;
                FOR asum += AI(DTYPE0);
                YPP = asum;
                NEXT
            }
        }
        else {
            const Py_ssize_t i_simd = LENGTH - LENGTH % 16;
            WHILE {
                float sum_simd, sum = 0.0;
                float *ad = (float *)it.pa;
                __m128 vsum0, vsum1, vsum2, vsum3;
                npy_uintp peel = calc_peel(ad, sizeof(float), 16);
                npy_intp i = 0;
                for (; i < peel; i++) {
                    sum += ad[i];
                }
                vsum0 = _mm_load_ps(&ad[peel + 0]);
                vsum1 = _mm_load_ps(&ad[peel + 4]);
                vsum2 = _mm_load_ps(&ad[peel + 8]);
                vsum3 = _mm_load_ps(&ad[peel + 12]);
                for (i = i + 16; i < i_simd + peel; i += 16)
                {
                    __m128 v0 = _mm_load_ps(&ad[i]);
                    __m128 v1 = _mm_load_ps(&ad[i + 4]);
                    __m128 v2 = _mm_load_ps(&ad[i + 8]);
                    __m128 v3 = _mm_load_ps(&ad[i + 12]);
                    vsum0 = _mm_add_ps(vsum0, v0);
                    vsum1 = _mm_add_ps(vsum1, v1);
                    vsum2 = _mm_add_ps(vsum2, v2);
                    vsum3 = _mm_add_ps(vsum3, v3);
                }
                vsum0 = _mm_add_ps(vsum0, vsum1);
                vsum1 = _mm_add_ps(vsum2, vsum3);
                vsum0 = _mm_add_ps(vsum0, vsum1);

                __m128 shuf = _mm_movehdup_ps(vsum0);
                __m128 sums = _mm_add_ps(vsum0, shuf);
                shuf        = _mm_movehl_ps(shuf, sums);
                sums        = _mm_add_ss(sums, shuf);
                sum_simd    = _mm_cvtss_f32(sums);

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
        if (LENGTH < 4) {
            WHILE {
                FOR {
                    YI(DTYPE0) += AI(DTYPE0);
                }
                NEXT2
            }
        }
        else {
            if (!IS_CONTIGUOUS(a) || LENGTH & 3 || (npy_uintp)it.pa & 15) {
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
                const Py_ssize_t i_simd = LENGTH - LENGTH % 16;
                WHILE {
                    float *ad = (float *)it.pa;
                    float *yd = (float *)it.py;
                    npy_intp i = 0;
                    for (; i < i_simd; i += 16)
                    {
                        __m128 a0 = _mm_load_ps(&ad[i]);
                        __m128 a1 = _mm_load_ps(&ad[i + 4]);
                        __m128 a2 = _mm_load_ps(&ad[i + 8]);
                        __m128 a3 = _mm_load_ps(&ad[i + 12]);

                        __m128 y0 = _mm_load_ps(&yd[i]);
                        __m128 y1 = _mm_load_ps(&yd[i + 4]);
                        __m128 y2 = _mm_load_ps(&yd[i + 8]);
                        __m128 y3 = _mm_load_ps(&yd[i + 12]);

                        _mm_store_ps(&yd[i],     _mm_add_ps(y0, a0));
                        _mm_store_ps(&yd[i + 4], _mm_add_ps(y1, a1));
                        _mm_store_ps(&yd[i + 8], _mm_add_ps(y2, a2));
                        _mm_store_ps(&yd[i + 12], _mm_add_ps(y3, a3));
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

/* dtype = [['int64'], ['int32']] */
static PyObject *
sum11_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                npy_DTYPE0 asum = 0;
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
sum11(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer02(args,
                     kwds,
                     sum11_float64,
                     sum11_float32,
                     sum11_int64,
                     sum11_int32);
}


/* sum12 ----------------------------------------------------------------- */

/* simd: avx */

/* dtype = [['float64']] */
static PyObject *
sum12_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 19 || !IS_CONTIGUOUS(a)) {
            /* could loop unroll here */
            WHILE {
                npy_DTYPE0 asum = 0;
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
sum12_DTYPE0(PyArrayObject *a, int axis, int fast_axis)
{
    PyObject *y;
    if (axis == fast_axis) {
        INIT01(DTYPE0, DTYPE0)
        if (LENGTH < 4) {
            WHILE {
                npy_DTYPE0 asum = 0;
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
sum12(PyObject *self, PyObject *args, PyObject *kwds)
{
    return reducer02(args,
                     kwds,
                     sum12_float64,
                     sum12_float32,
                     sum12_int64,
                     sum12_int32);
}


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

static char module_doc[] = "femto's some sums.";

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
    {"sum00",   (PyCFunction)sum00,   VARKEY, sum_doc},
    {"sum01",   (PyCFunction)sum01,   VARKEY, sum_doc},
    {"p_sum01", (PyCFunction)p_sum01, VARKEY, sum_doc},
    {"sum02",   (PyCFunction)sum02,   VARKEY, sum_doc},
    {"p_sum02", (PyCFunction)p_sum02, VARKEY, sum_doc},
    {"sum03",   (PyCFunction)sum03,   VARKEY, sum_doc},
    {"p_sum03", (PyCFunction)p_sum03, VARKEY, sum_doc},
    {"sum04",   (PyCFunction)sum04,   VARKEY, sum_doc},
    {"p_sum04", (PyCFunction)p_sum04, VARKEY, sum_doc},
    {"sum10",   (PyCFunction)sum10,   VARKEY, sum_doc},
    {"sum11",   (PyCFunction)sum11,   VARKEY, sum_doc},
    {"sum12",   (PyCFunction)sum12,   VARKEY, sum_doc},
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
