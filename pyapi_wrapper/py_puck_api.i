%module(directors="1") py_puck_api

using namespace std;

%{
#define SWIG_FILE_WITH_INIT
#include "pyapi_wrapper/py_api_wrapper.h"
    using namespace py_puck_api;
    using namespace puck;
%}

%include "pyapi_wrapper/numpy.i"

%init %{
import_array();
%}
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef signed long long int int64_t;
typedef unsigned long long int uint64_t;
typedef long time_t;
typedef long off_t;
typedef unsigned int  dev_t;
typedef uint32_t gid_t;
typedef unsigned int mode_t;
typedef unsigned int uid_t;

%{
PyObject *swig_ptr (PyObject *a)
{

    if (PyBytes_Check(a)) {
        return SWIG_NewPointerObj(PyBytes_AsString(a), SWIGTYPE_p_char, 0);
    }
    if (PyByteArray_Check(a)) {
        return SWIG_NewPointerObj(PyByteArray_AsString(a), SWIGTYPE_p_char, 0);
    }
    if(!PyArray_Check(a)) {
        PyErr_SetString(PyExc_ValueError, "input not a numpy array");
        return NULL;
    }
    PyArrayObject *ao = (PyArrayObject *)a;

    if(!PyArray_ISCONTIGUOUS(ao)) {
        PyErr_SetString(PyExc_ValueError, "array is not C-contiguous");
        return NULL;
    }
    void * data = PyArray_DATA(ao);
    if(PyArray_TYPE(ao) == NPY_FLOAT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_float, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT64) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_double, 0);
    }
    if(PyArray_TYPE(ao) == NPY_FLOAT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT8) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_char, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT16) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_short, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_INT32) {
        return SWIG_NewPointerObj(data, SWIGTYPE_p_int, 0);
    }
    if(PyArray_TYPE(ao) == NPY_UINT64) {
#ifdef SWIGWORDSIZE64
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_unsigned_long_long, 0);
#endif
    }
    if(PyArray_TYPE(ao) == NPY_INT64) {
#ifdef SWIGWORDSIZE64
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long, 0);
#else
        return SWIG_NewPointerObj(data, SWIGTYPE_p_long_long, 0);
#endif
    }
    PyErr_SetString(PyExc_ValueError, "did not recognize array type");
    return NULL;
}




%}

// return a pointer usable as input for functions that expect pointers
PyObject *swig_ptr (PyObject *a);

%define REV_SWIG_PTR(ctype, numpytype)

%{
PyObject * rev_swig_ptr(ctype *src, npy_intp size) {
    return PyArray_SimpleNewFromData(1, &size, numpytype, src);
}
%}

PyObject * rev_swig_ptr(ctype *src, size_t size);

%enddef
REV_SWIG_PTR(float, NPY_FLOAT32);
REV_SWIG_PTR(double, NPY_FLOAT64);
REV_SWIG_PTR(unsigned char, NPY_UINT8);
REV_SWIG_PTR(char, NPY_INT8);
REV_SWIG_PTR(unsigned short, NPY_UINT16);
REV_SWIG_PTR(short, NPY_INT16);
REV_SWIG_PTR(int, NPY_INT32);
REV_SWIG_PTR(unsigned int, NPY_UINT32);
REV_SWIG_PTR(int64_t, NPY_INT64);
REV_SWIG_PTR(uint64_t, NPY_UINT64);


namespace py_puck_api{
    void update_gflag(const char* gflag_key, const char* gflag_val);
    class PySearcher{
        public:
            PySearcher();
            int build(uint32_t n);
            void show();
            int init();
            void search(uint32_t n, const float* x,const uint32_t topk, float* distances, uint32_t* labels);
            void batch_add(uint32_t n, uint32_t dim, const float* features, const uint32_t* labels);
            void batch_delete(uint32_t n, const uint32_t* labels);
            ~PySearcher();
    };
};
