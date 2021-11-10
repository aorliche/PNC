
// Reshape a matrix to get rid of duplicate entries for i.e. Pearson or cdist

// Compile with:
// cl /LD /I"C:/Program Files/Python38/include" /I"C:\Program Files\Python38\Lib\site-packages\numpy\core\include\numpy" Reshape.c "C:/Program Files/Python38/libs/python38.lib"
// cl /LD /I"C:/Users/aorlichenko/AppData/Local/Programs/Python/Python39/include" /I"C:/Users/aorlichenko/AppData/Local/Programs/Python/Python39/Lib/site-packages/numpy/core/include/numpy" Reshape.c "C:/Users/aorlichenko/AppData/Local/Programs/Python/Python39/libs/python39.lib"

// Don't forget to move Reshape.dll to Reshape.pyd!

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "ndarrayobject.h"
#include "math.h"

static PyObject *ReshapeError;

#define GET_NENT(nr) (nr*(nr+1)/2)

static PyObject *
Reshape_Reshape(PyObject *self, PyObject *args) {
    PyObject *o1, *o2;
    PyArrayObject *ao1, *ao2;
    
    if (!PyArg_ParseTuple(args, "OO", &o1, &o2)) {
		PyErr_SetString(ReshapeError, "Usage: Reshape(mat1, mat2)");
        return NULL;
    }
    
    // Check that we're passed ndarrays
    if (!PyArray_Check(o1)) {
        PyErr_SetString(ReshapeError, "ndarray expected as first argument");
        return NULL;
    }
    if (!PyArray_Check(o2)) {
        PyErr_SetString(ReshapeError, "ndarray expected as second argument");
        return NULL;
    }
    
    ao1 = (PyArrayObject *)o1;
    ao2 = (PyArrayObject *)o2;
    
    // Check that first arg has four dimensions
    if (ao1->nd != 4) {
        PyErr_SetString(ReshapeError, "first arg ndarray must be NSxNRxNRxNd");
        return NULL;
    }
    const int ns = ao1->dimensions[0];
    const int nr = ao1->dimensions[1];
	const int nr2 = ao1->dimensions[2];
    const int nd = ao1->dimensions[3];
	
	// Equal second and third dimensions
	if (nr != nr2) {
		PyErr_SetString(ReshapeError, "first arg ndarray must be NSxNRxNRxNd");
        return NULL;
	}
    
    // Check that second arg has three dimensions
    if (ao2->nd != 3) {
        PyErr_SetString(ReshapeError, "second arg ndarray must be NSx(NR*(NR+1)/2)xNd");
        return NULL;
    }
    
    // Check that number of subjects dimensions are equal
    if (ao1->dimensions[0] != ao2->dimensions[0]) {
        PyErr_SetString(ReshapeError, "number of subjects (first dimension of both arguments) must be the same");
        return NULL;
    }
	
	// Check that number of distance dimensions are equal
    if (ao1->dimensions[3] != ao2->dimensions[2]) {
        PyErr_SetString(ReshapeError, "number of distances must be equal");
        return NULL;
    }
    
    // Check correct size of NR 
    int nc = GET_NENT(nr);
    if (ao2->dimensions[1] != nc) {
        char buf[200];
        sprintf(buf, "second arg must be of size %dx%dx%d for first arg of size %dx%dx%dx%d", ns, nc, nd, ns, nr, nr, nd);
        PyErr_SetString(ReshapeError, buf);
        return NULL;
    }
    
    // Copy data
    for (int n=0; n<ns; n++) {
		int c = 0;
        for (int i=0; i<nr; i++) {
            for (int j=0; j<nr; j++) {
				if (j < i) continue;
                for (int k=0; k<nd; k++) {
					float *from = PyArray_GETPTR4(ao1, n, i, j, k);
					float *to = PyArray_GETPTR3(ao2, n, c, k);
					*to = *from;
                }
				c++;
            }
        }
    }
    
    Py_INCREF(ao2);
    
    return (PyObject *)ao2;
}
    
static PyMethodDef ReshapeMethods[] = {
    {"Reshape", Reshape_Reshape, METH_VARARGS, "Reshaping getting rid of redundant data to reduce space"},
    {NULL, NULL, NULL}
};

static struct PyModuleDef ReshapeModule = {
    PyModuleDef_HEAD_INIT,
    "Reshape",
    NULL,
    -1,
    ReshapeMethods
};

PyMODINIT_FUNC
PyInit_Reshape(void)
{
    PyObject *m;
    
    // https://stackoverflow.com/questions/47026900/pyarray-check-gives-segmentation-fault-with-cython-c
    import_array();
    
    m = PyModule_Create(&ReshapeModule);
    if (m == NULL) {
        return NULL;
    }
    
    ReshapeError = PyErr_NewException("Reshape.error", NULL, NULL);
    Py_XINCREF(ReshapeError);
    if (PyModule_AddObject(m, "error", ReshapeError) < 0) {
        Py_XDECREF(ReshapeError);
        Py_CLEAR(ReshapeError);
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
}

