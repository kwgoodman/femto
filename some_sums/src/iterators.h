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

#include <numpy/arrayobject.h>

/*
   some_sums iterators are based on ideas from NumPy's PyArray_IterAllButAxis
   and PyArray_ITER_NEXT.
*/

struct _iter {
    int        ndim_m2; /* ndim - 2 */
    int        axis;    /* axis to not iterate over */
    Py_ssize_t length;  /* a.shape[axis] */
    Py_ssize_t astride; /* a.strides[axis] */
    npy_intp   i;       /* integer used by some macros */
    npy_intp   its;     /* number of iterations completed */
    npy_intp   nits;    /* number of iterations iterator plans to make */
    npy_intp   indices[NPY_MAXDIMS];  /* current location of iterator */
    npy_intp   astrides[NPY_MAXDIMS]; /* a.strides, a.strides[axis] removed */
    npy_intp   shape[NPY_MAXDIMS];    /* a.shape, a.shape[axis] removed */
    char       *pa;     /* pointer to data corresponding to indices */
};
typedef struct _iter iter;

static BN_INLINE void
init_iter(iter *it, PyArrayObject *a, int axis)
{
    int i, j = 0;
    const int ndim = PyArray_NDIM(a);
    const npy_intp *shape = PyArray_SHAPE(a);
    const npy_intp *strides = PyArray_STRIDES(a);

    it->axis = axis;
    it->its = 0;
    it->nits = 1;
    it->ndim_m2 = ndim - 2;
    it->pa = PyArray_BYTES(a);

    for (i = 0; i < ndim; i++) {
        if (i == axis) {
            it->astride = strides[i];
            it->length = shape[i];
        }
        else {
            it->indices[j] = 0;
            it->astrides[j] = strides[i];
            it->shape[j] = shape[i];
            it->nits *= shape[i];
            j++;
        }
    }
}

#define NEXT \
    for (it.i = it.ndim_m2; it.i > -1; it.i--) { \
        if (it.indices[it.i] < it.shape[it.i] - 1) { \
            it.pa += it.astrides[it.i]; \
            it.indices[it.i]++; \
            break; \
        } \
        it.pa -= it.indices[it.i] * it.astrides[it.i]; \
        it.indices[it.i] = 0; \
    } \
    it.its++;

/* macros used with iterators -------------------------------------------- */

/* most of these macros assume iterator is named `it` */

#define  NDIM           it.ndim_m2 + 2
#define  SHAPE          it.shape
#define  LENGTH         it.length

#define  WHILE          while (it.its < it.nits)
#define  FOR            for (it.i = 0; it.i < it.length; it.i++)

#define  AI(dtype)      *(npy_##dtype *)(it.pa + it.i * it.astride)
#define  AX(dtype, x)   *(npy_##dtype *)(it.pa + (x) * it.astride)

#define  YPP            *py++
#define  FILL_Y(value) \
    int i; \
    Py_ssize_t size = PyArray_SIZE((PyArrayObject *)y); \
    for (i = 0; i < size; i++) YPP = value;
