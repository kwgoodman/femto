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

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include <numpy/arrayobject.h>

/* for ease of dtype templating */
#define NPY_float64 NPY_FLOAT64
#define NPY_float32 NPY_FLOAT32
#define NPY_int64   NPY_INT64
#define NPY_int32   NPY_INT32
#define NPY_intp    NPY_INTP
#define NPY_MAX_int64 NPY_MAX_INT64
#define NPY_MAX_int32 NPY_MAX_INT32
#define NPY_MIN_int64 NPY_MIN_INT64
#define NPY_MIN_int32 NPY_MIN_INT32

#if PY_MAJOR_VERSION >= 3
    #define PyString_FromString PyBytes_FromString
    #define PyInt_FromLong PyLong_FromLong
    #define PyInt_AsLong PyLong_AsLong
    #define PyString_InternFromString PyUnicode_InternFromString
#endif

#define VARKEY METH_VARARGS | METH_KEYWORDS
#define error_converting(x) (((x) == -1) && PyErr_Occurred())

#define VALUE_ERR(text)   PyErr_SetString(PyExc_ValueError, text)
#define TYPE_ERR(text)    PyErr_SetString(PyExc_TypeError, text)
#define MEMORY_ERR(text)  PyErr_SetString(PyExc_MemoryError, text)
#define RUNTIME_ERR(text) PyErr_SetString(PyExc_RuntimeError, text)

/* `inline` copied from NumPy. */
#if defined(_MSC_VER)
        #define BN_INLINE __inline
#elif defined(__GNUC__)
	#if defined(__STRICT_ANSI__)
		#define BN_INLINE __inline__
	#else
		#define BN_INLINE inline
	#endif
#else
        #define BN_INLINE
#endif

#define C_CONTIGUOUS(a) PyArray_CHKFLAGS(a, NPY_ARRAY_C_CONTIGUOUS)
#define F_CONTIGUOUS(a) PyArray_CHKFLAGS(a, NPY_ARRAY_F_CONTIGUOUS)
#define IS_CONTIGUOUS(a) (C_CONTIGUOUS(a) || F_CONTIGUOUS(a))
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_11_API_VERSION
#include <numpy/arrayobject.h>

/* for ease of dtype templating */
#define NPY_float64 NPY_FLOAT64
#define NPY_float32 NPY_FLOAT32
#define NPY_int64   NPY_INT64
#define NPY_int32   NPY_INT32
#define NPY_intp    NPY_INTP
#define NPY_MAX_int64 NPY_MAX_INT64
#define NPY_MAX_int32 NPY_MAX_INT32
#define NPY_MIN_int64 NPY_MIN_INT64
#define NPY_MIN_int32 NPY_MIN_INT32

#if PY_MAJOR_VERSION >= 3
    #define PyString_FromString PyBytes_FromString
    #define PyInt_FromLong PyLong_FromLong
    #define PyInt_AsLong PyLong_AsLong
    #define PyString_InternFromString PyUnicode_InternFromString
#endif

#define VARKEY METH_VARARGS | METH_KEYWORDS
#define error_converting(x) (((x) == -1) && PyErr_Occurred())

#define VALUE_ERR(text)   PyErr_SetString(PyExc_ValueError, text)
#define TYPE_ERR(text)    PyErr_SetString(PyExc_TypeError, text)
#define MEMORY_ERR(text)  PyErr_SetString(PyExc_MemoryError, text)
#define RUNTIME_ERR(text) PyErr_SetString(PyExc_RuntimeError, text)

/* `inline` copied from NumPy. */
#if defined(_MSC_VER)
        #define BN_INLINE __inline
#elif defined(__GNUC__)
	#if defined(__STRICT_ANSI__)
		#define BN_INLINE __inline__
	#else
		#define BN_INLINE inline
	#endif
#else
        #define BN_INLINE
#endif

#define C_CONTIGUOUS(a) PyArray_CHKFLAGS(a, NPY_ARRAY_C_CONTIGUOUS)
#define F_CONTIGUOUS(a) PyArray_CHKFLAGS(a, NPY_ARRAY_F_CONTIGUOUS)
#define IS_CONTIGUOUS(a) (C_CONTIGUOUS(a) || F_CONTIGUOUS(a))

#define INIT(dtype0, dtype1) \
    iter it; \
    PyObject *y; \
    npy_##dtype1 *py; \
    init_iter(&it, a, axis); \
    y = PyArray_EMPTY(NDIM - 1, SHAPE, NPY_##dtype0, 0); \
    py = (npy_##dtype1 *)PyArray_DATA((PyArrayObject *)y);

#define INIT2(dtype0, dtype1) \
    iter2 it; \
    PyObject *y; \
    int i, j=0, ndim = PyArray_NDIM(a); \
    npy_intp shape[NPY_MAXDIMS]; \
    for (i = 0; i < ndim; i++) { \
        if (i != axis) { \
            shape[j++] = PyArray_DIM(a, i); \
        }  \
    } \
    y = PyArray_ZEROS(ndim - 1, shape, NPY_##dtype0, 0); \
    init_iter2(&it, a, y, axis, min_axis); \

#define REDUCE(name, dtype) \
    static PyObject * \
    name##_##dtype(PyArrayObject *a, int axis)

#define REDUCE_MAIN(name) \
    static PyObject * \
    name(PyObject *self, PyObject *args, PyObject *kwds) \
    { \
        return reducer(args, \
                       kwds, \
                       name##_float64, \
                       name##_float32, \
                       name##_int64, \
                       name##_int32); \
    }

typedef PyObject *(*fone_t)(PyArrayObject *a, int axis);
typedef PyObject *(*fnf_t)(PyArrayObject *a, int axis, int min_axis);

static PyObject *
reducer(PyObject *args,
        PyObject *kwds,
        fone_t fone_float64,
        fone_t fone_float32,
        fone_t fone_int64,
        fone_t fone_int32);

static PyObject *
reducer02(PyObject *args,
          PyObject *kwds,
          fone_t fone_float64,
          fone_t fone_float32,
          fone_t fone_int64,
          fone_t fone_int32,
          fnf_t f_float64,
          fnf_t f_float32,
          fnf_t f_int64,
          fnf_t f_int32);
