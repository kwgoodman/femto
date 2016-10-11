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

/* sum00 ----------------------------------------------------------------- */

/* simple for loop */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
REDUCE(sum00, DTYPE0)
{
    npy_DTYPE0 asum;
    INIT(DTYPE0, DTYPE0)
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR asum += AI(DTYPE0);
            YPP = asum;
            NEXT
        }
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
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
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
    return y;
}
/* dtype end */

REDUCE_MAIN(sum01)


/* sum02 ----------------------------------------------------------------- */

/* add special casing for summing along non-fast axis */

/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
static PyObject *
sum02_DTYPE0(PyArrayObject *a, int axis, int min_axis)
{
    INIT2(DTYPE0, DTYPE0)
    if (LENGTH == 0) {
        char *py = PyArray_BYTES((PyArrayObject *)y);
        FILL_Y(0)
    }
    else {
        //memset(it.py, 0, PyArray_SIZE((PyArrayObject *)y) * sizeof(npy_DTYPE0));
        WHILE {
            FOR {
                *(npy_DTYPE0 *)(it.py + it.i * it.ystride) += AI(DTYPE0);
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
                     sum01_float64,
                     sum01_float32,
                     sum01_int64,
                     sum01_int32,
                     sum02_float64,
                     sum02_float32,
                     sum02_int64,
                     sum02_int32);
}


/* sum03, sum04 ---------------------------------------------------------- */

/* repeat = {'NAME': ['sum03', 'sum04'],
             'PREFETCH': ['', 
             '__builtin_prefetch(it.pa + i * it.astride + 1024, 0, 3);']} */
/* dtype = [['float64'], ['float32'], ['int64'], ['int32']] */
REDUCE(NAME, DTYPE0)
{
    npy_DTYPE0 asum;
    INIT(DTYPE0, DTYPE0)
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        if (LENGTH < 8) {
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
                Py_ssize_t repeat = LENGTH - LENGTH % 8;
                npy_DTYPE0 s[8];
                s[0] = AX(DTYPE0, 0);
                s[1] = AX(DTYPE0, 1);
                s[2] = AX(DTYPE0, 2);
                s[3] = AX(DTYPE0, 3);
                s[4] = AX(DTYPE0, 4);
                s[5] = AX(DTYPE0, 5);
                s[6] = AX(DTYPE0, 6);
                s[7] = AX(DTYPE0, 7);
                for (i = 8; i < repeat; i += 8) {
                    PREFETCH
                    s[0] += AX(DTYPE0, i);
                    s[1] += AX(DTYPE0, i + 1);
                    s[2] += AX(DTYPE0, i + 2);
                    s[3] += AX(DTYPE0, i + 3);
                    s[4] += AX(DTYPE0, i + 4);
                    s[5] += AX(DTYPE0, i + 5);
                    s[6] += AX(DTYPE0, i + 6);
                    s[7] += AX(DTYPE0, i + 7);
                }
                for (i = i; i < LENGTH; i++) {
                    s[0] += AX(DTYPE0, i);
                }
                YPP = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];
                NEXT
            }
        }
    }
    return y;
}
/* dtype end */

REDUCE_MAIN(NAME)
/* repeat end */


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
          fone_t ffast_float64,
          fone_t ffast_float32,
          fone_t ffast_int64,
          fone_t ffast_int32,
          fnf_t f_float64,
          fnf_t f_float32,
          fnf_t f_int64,
          fnf_t f_int32)
{

    int ndim;
    int axis;
    int dtype;
    int fast;
    int min_axis;

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

    fast = 0;
    if (C_CONTIGUOUS(a) && axis == ndim - 1) {
        fast = 1;
        min_axis = axis;
    }
    else if (F_CONTIGUOUS(a) && axis == 0) {
        fast = 1;
        min_axis = axis;
    }
    else {
        int i;
        min_axis = 0;
        npy_intp *strides = PyArray_STRIDES(a);
        npy_intp min_stride = strides[0];
        for (i = 1; i < ndim; i++) {
            if (strides[i] < min_stride) {
                min_stride = strides[i];
                min_axis = i;
            }
        }
        if (min_axis == axis) {
            fast = 1;
        }
    }

    dtype = PyArray_TYPE(a);

    if (fast) {
        if (dtype == NPY_FLOAT64) {
            return ffast_float64(a, axis);
        }
        else if (dtype == NPY_FLOAT32) {
            return ffast_float32(a, axis);
        }
        else if (dtype == NPY_INT64) {
            return ffast_int64(a, axis);
        }
        else if (dtype == NPY_INT32) {
            return ffast_int32(a, axis);
        }
        else {
            return PyArray_Sum(a, axis, dtype, NULL);
        }
    }
    else {
        if (dtype == NPY_FLOAT64) {
            return f_float64(a, axis, min_axis);
        }
        else if (dtype == NPY_FLOAT32) {
            return f_float32(a, axis, min_axis);
        }
        else if (dtype == NPY_INT64) {
            return f_int64(a, axis, min_axis);
        }
        else if (dtype == NPY_INT32) {
            return f_int32(a, axis, min_axis);
        }
        else {
            return PyArray_Sum(a, axis, dtype, NULL);
        }
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
