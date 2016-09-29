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

#include "some_sums.h"
#include "iterators.h"

/* init macros ----------------------------------------------------------- */

#define INIT_ALL \
    iter it; \
    init_iter_all(&it, a, 0);

#define INIT_ALL_RAVEL \
    iter it; \
    init_iter_all(&it, a, 1);

#define INIT_ONE(dtype0, dtype1) \
    iter it; \
    PyObject *y; \
    npy_##dtype1 *py; \
    init_iter_one(&it, a, axis); \
    y = PyArray_EMPTY(NDIM - 1, SHAPE, NPY_##dtype0, 0); \
    py = (npy_##dtype1 *)PyArray_DATA((PyArrayObject *)y);

/* function signatures --------------------------------------------------- */

/* low-level functions such as nansum_all_float64 */
#define REDUCE_ALL(name, dtype) \
    static PyObject * \
    name##_all_##dtype(PyArrayObject *a, int ddof)

/* low-level functions such as nansum_one_float64 */
#define REDUCE_ONE(name, dtype) \
    static PyObject * \
    name##_one_##dtype(PyArrayObject *a, int axis, int ddof)

/* top-level functions such as nansum */
#define REDUCE_MAIN(name, ravel, has_ddof) \
    static PyObject * \
    name(PyObject *self, PyObject *args, PyObject *kwds) \
    { \
        return reducer(#name, \
                       args, \
                       kwds, \
                       name##_all_float64, \
                       name##_all_float32, \
                       name##_all_int64, \
                       name##_all_int32, \
                       name##_one_float64, \
                       name##_one_float32, \
                       name##_one_int64, \
                       name##_one_int32, \
                       ravel, \
                       has_ddof); \
    }

/* typedefs and prototypes ----------------------------------------------- */

typedef PyObject *(*fall_t)(PyArrayObject *a, int ddof);
typedef PyObject *(*fone_t)(PyArrayObject *a, int axis, int ddof);

static PyObject *
reducer(char *name,
        PyObject *args,
        PyObject *kwds,
        fall_t fall_float64,
        fall_t fall_float32,
        fall_t fall_int64,
        fall_t fall_int32,
        fone_t fone_float64,
        fone_t fone_float32,
        fone_t fone_int64,
        fone_t fone_int32,
        int ravel,
        int has_ddof);

/* nansum ---------------------------------------------------------------- */

/* dtype = [['float64'], ['float32']] */
REDUCE_ALL(nansum, DTYPE0)
{
    npy_DTYPE0 ai, asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR {
            ai = AI(DTYPE0);
            if (ai == ai) asum += ai;
        }
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyFloat_FromDouble(asum);
}

REDUCE_ONE(nansum, DTYPE0)
{
    npy_DTYPE0 ai, asum;
    INIT_ONE(DTYPE0, DTYPE0)
    BN_BEGIN_ALLOW_THREADS
    if (LENGTH == 0) {
        FILL_Y(0)
    }
    else {
        WHILE {
            asum = 0;
            FOR {
                ai = AI(DTYPE0);
                if (ai == ai) asum += ai;
            }
            YPP = asum;
            NEXT
        }
    }
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

/* dtype = [['int64'], ['int32']] */
REDUCE_ALL(nansum, DTYPE0)
{
    npy_DTYPE0 asum = 0;
    INIT_ALL
    BN_BEGIN_ALLOW_THREADS
    WHILE {
        FOR asum += AI(DTYPE0);
        NEXT
    }
    BN_END_ALLOW_THREADS
    return PyInt_FromLong(asum);
}

REDUCE_ONE(nansum, DTYPE0)
{
    npy_DTYPE0 asum;
    INIT_ONE(DTYPE0, DTYPE0)
    BN_BEGIN_ALLOW_THREADS
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
    BN_END_ALLOW_THREADS
    return y;
}
/* dtype end */

REDUCE_MAIN(nansum, 0, 0)

/* python strings -------------------------------------------------------- */

PyObject *pystr_a = NULL;
PyObject *pystr_axis = NULL;
PyObject *pystr_ddof = NULL;

static int
intern_strings(void) {
    pystr_a = PyString_InternFromString("a");
    pystr_axis = PyString_InternFromString("axis");
    pystr_ddof = PyString_InternFromString("ddof");
    return pystr_a && pystr_axis && pystr_ddof;
}

/* reducer --------------------------------------------------------------- */

static BN_INLINE int
parse_args(PyObject *args,
           PyObject *kwds,
           int has_ddof,
           PyObject **a,
           PyObject **axis,
           PyObject **ddof)
{
    const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    const Py_ssize_t nkwds = kwds == NULL ? 0 : PyDict_Size(kwds);
    if (nkwds) {
        int nkwds_found = 0;
        PyObject *tmp;
        switch (nargs) {
            case 2:
                if (has_ddof) {
                    *axis = PyTuple_GET_ITEM(args, 1);
                } else {
                    TYPE_ERR("wrong number of arguments");
                    return 0;
                }
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
            case 2:
                if (has_ddof) {
                    tmp = PyDict_GetItem(kwds, pystr_ddof);
                    if (tmp != NULL) {
                        *ddof = tmp;
                        nkwds_found++;
                    }
                    break;
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
        if (nargs + nkwds_found > 2 + has_ddof) {
            TYPE_ERR("too many arguments");
            return 0;
        }
    }
    else {
        switch (nargs) {
            case 3:
                if (has_ddof) {
                    *ddof = PyTuple_GET_ITEM(args, 2);
                } else {
                    TYPE_ERR("wrong number of arguments");
                    return 0;
                }
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
reducer(char *name,
        PyObject *args,
        PyObject *kwds,
        fall_t fall_float64,
        fall_t fall_float32,
        fall_t fall_int64,
        fall_t fall_int32,
        fone_t fone_float64,
        fone_t fone_float32,
        fone_t fone_int64,
        fone_t fone_int32,
        int ravel,
        int has_ddof)
{

    int ndim;
    int axis;
    int dtype;
    int ddof;
    int reduce_all = 0;

    PyArrayObject *a;

    PyObject *a_obj = NULL;
    PyObject *axis_obj = Py_None;
    PyObject *ddof_obj = NULL;

    if (!parse_args(args, kwds, has_ddof, &a_obj, &axis_obj, &ddof_obj)) {
        return NULL;
    }

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
        return slow(name, args, kwds);
    }

    /* does user want to reduce over all axes? */
    if (axis_obj == Py_None) {
        reduce_all = 1;
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
            reduce_all = 1;
        }
    }

    /* ddof */
    if (ddof_obj == NULL) {
        ddof = 0;
    }
    else {
        ddof = PyArray_PyIntAsInt(ddof_obj);
        if (error_converting(ddof)) {
            TYPE_ERR("`ddof` must be an integer");
            return NULL;
        }
    }

    dtype = PyArray_TYPE(a);

    if (reduce_all == 1) {
        /* we are reducing the array along all axes */
        if (dtype == NPY_FLOAT64) {
            return fall_float64(a, ddof);
        }
        else if (dtype == NPY_FLOAT32) {
            return fall_float32(a, ddof);
        }
        else if (dtype == NPY_INT64) {
            return fall_int64(a, ddof);
        }
        else if (dtype == NPY_INT32) {
            return fall_int32(a, ddof);
        }
        else {
            return slow(name, args, kwds);
        }
    }
    else {
        /* we are reducing an array with ndim > 1 over a single axis */
        if (dtype == NPY_FLOAT64) {
            return fone_float64(a, axis, ddof);
        }
        else if (dtype == NPY_FLOAT32) {
            return fone_float32(a, axis, ddof);
        }
        else if (dtype == NPY_INT64) {
            return fone_int64(a, axis, ddof);
        }
        else if (dtype == NPY_INT32) {
            return fone_int32(a, axis, ddof);
        }
        else {
            return slow(name, args, kwds);
        }

    }

}

/* docstrings ------------------------------------------------------------- */

static char reduce_doc[] = "some_sums's some sums.";

static char nansum_doc[] =
/* MULTILINE STRING BEGIN
nansum(a, axis=None)

Sum of array elements along given axis treating NaNs as zero.

The data type (dtype) of the output is the same as the input. On 64-bit
operating systems, 32-bit input is NOT upcast to 64-bit accumulator and
return values.

Parameters
----------
a : array_like
    Array containing numbers whose sum is desired. If `a` is not an
    array, a conversion is attempted.
axis : {int, None}, optional
    Axis along which the sum is computed. The default (axis=None) is to
    compute the sum of the flattened array.

Returns
-------
y : ndarray
    An array with the same shape as `a`, with the specified axis removed.
    If `a` is a 0-d array, or if axis is None, a scalar is returned.

Notes
-----
No error is raised on overflow.

If positive or negative infinity are present the result is positive or
negative infinity. But if both positive and negative infinity are present,
the result is Not A Number (NaN).

Examples
--------
>>> ss.nansum(1)
1
>>> ss.nansum([1])
1
>>> ss.nansum([1, np.nan])
1.0
>>> a = np.array([[1, 1], [1, np.nan]])
>>> ss.nansum(a)
3.0
>>> ss.nansum(a, axis=0)
array([ 2.,  1.])

When positive infinity and negative infinity are present:

>>> ss.nansum([1, np.nan, np.inf])
inf
>>> ss.nansum([1, np.nan, np.NINF])
-inf
>>> ss.nansum([1, np.nan, np.inf, np.NINF])
nan

MULTILINE STRING END */

/* python wrapper -------------------------------------------------------- */

static PyMethodDef
reduce_methods[] = {
    {"nansum",    (PyCFunction)nansum,    VARKEY, nansum_doc},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef
reduce_def = {
   PyModuleDef_HEAD_INIT,
   "reduce",
   reduce_doc,
   -1,
   reduce_methods
};
#endif


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
#define RETVAL m
PyInit_reduce(void)
#else
#define RETVAL
initreduce(void)
#endif
{
    #if PY_MAJOR_VERSION >=3
        PyObject *m = PyModule_Create(&reduce_def);
    #else
        PyObject *m = Py_InitModule3("reduce", reduce_methods, reduce_doc);
    #endif
    if (m == NULL) return RETVAL;
    import_array();
    if (!intern_strings()) {
        return RETVAL;
    }
    return RETVAL;
}
