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

#ifndef SOME_SUMS_H
#define SOME_SUMS_H

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

/* slow ------------------------------------------------------------------ */

static PyObject *slow_module = NULL;

static PyObject *
slow(char *name, PyObject *args, PyObject *kwds)
{
    PyObject *func = NULL;
    PyObject *out = NULL;

    if (slow_module == NULL) {
        /* some_sums.slow has not been imported during the current
         * python session. Only import it once per session to save time */
        slow_module = PyImport_ImportModule("some_sums.slow");
        if (slow_module == NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Cannot import some_sums.slow");
            return NULL;
        }
    }

    func = PyObject_GetAttrString(slow_module, name);
    if (func == NULL) {
        PyErr_Format(PyExc_RuntimeError,
                     "Cannot import %s from some_sums.slow", name);
        return NULL;
    }
    if (PyCallable_Check(func)) {
        out = PyObject_Call(func, args, kwds);
        if (out == NULL) {
            Py_XDECREF(func);
            return NULL;
        }
    }
    else {
        Py_XDECREF(func);
        PyErr_Format(PyExc_RuntimeError,
                     "some_sums.slow.%s is not callable", name);
        return NULL;
    }
    Py_XDECREF(func);

    return out;
}

#endif /* SOME_SUMS_H */
