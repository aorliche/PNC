
// Fast and space-efficient 3-way correlation function
// For 466 subjects and 264 ROIs, naive 3-way correlation takes (4*8)+GB
// This should reduce that number by up to 6 times (5.93 times)

// Compile with:
// cl /LD /I"C:/Program Files/Python38/include" /I"C:\Program Files\Python38\Lib\site-packages\numpy\core\include\numpy" ThreeWay.c "C:/Program Files/Python38/libs/python38.lib"
// cl /LD /I"C:/Users/aorlichenko/AppData/Local/Programs/Python/Python39/include" /I"C:/Users/aorlichenko/AppData/Local/Programs/Python/Python39/Lib/site-packages/numpy/core/include/numpy" ThreeWay.c "C:/Users/aorlichenko/AppData/Local/Programs/Python/Python39/libs/python39.lib"

// Need to use float numpy array for correlation argument!

// Don't forget to move ThreeWay.dll to ThreeWay.pyd!

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "ndarrayobject.h"
#include "math.h"
#include <thread>

static PyObject *ThreeWayError;

static int numEntries(int nroi) {
    int m1 = nroi-1;
    return (int)round(pow(m1,3)/6+pow(m1,2)+(11.0/6*m1)+1);
}

#define NCORE 6

// #define EPS_STEP 0.01   // 200 thresholds
// #define NEPS ((int)round(2/EPS_STEP))
// #define BING(i,j) binGraph[N*i+j]
// #define STR(FOO) #FOO

// Enable multithreading
// static void calcDist(int nroi, int *hist, int *dist) {
	// for (int i = nroi-1, a = 1, b = 2; i >= 0; i--) {
		// hist[i] = a;
		// a += b;
		// b += 1;
	// }
	// dist[0] = 0;
	// for (int i=1; i<nroi; i++) {
		// dist[i] = dist[i-1] + hist[i-1];
	// }
// }

// The multithreading function
static void calcCorr(int ns, int ne, int nr, int nt, PyArrayObject *aots, PyArrayObject *aocor) {
	for (int n=ns; n<ne; n++) {
		int c = 0;
		for (int i=0; i<nr; i++) {
			for (int j=i; j<nr; j++) {
				for (int k=j; k<nr; k++) {
					float cor = 0;
					for (int t=0; t<nt; t++) {
						float x1 = *(float *)PyArray_GETPTR3(aots, n, i, t);
						float x2 = *(float *)PyArray_GETPTR3(aots, n, j, t);
						float x3 = *(float *)PyArray_GETPTR3(aots, n, k, t);
						cor += x1*x2*x3;
					}
					float *ptr = (float *)PyArray_GETPTR2(aocor, n, c++);
					*ptr = cor;
				}
			}
		}
	}
}

static PyObject *
ThreeWay_ThreeWay(PyObject *self, PyObject *args) {
    PyObject *ots, *ocor;
    PyArrayObject *aots, *aocor;
    
    if (!PyArg_ParseTuple(args, "OO", &ots, &ocor)) {
        return NULL;
    }
    
    // Check that we're passed ndarrays
    if (!PyArray_Check(ots)) {
        PyErr_SetString(ThreeWayError, "ndarray expected as first argument");
        return NULL;
    }
    if (!PyArray_Check(ocor)) {
        PyErr_SetString(ThreeWayError, "ndarray expected as second argument");
        return NULL;
    }
    
    aots = (PyArrayObject *)ots;
    aocor = (PyArrayObject *)ocor;
    
    // Check that first arg has three dimensions
    if (aots->nd != 3) {
        PyErr_SetString(ThreeWayError, "first arg ndarray must be NSxNRxNt");
        return NULL;
    }
    const int ns = aots->dimensions[0];
    const int nr = aots->dimensions[1];
    const int nt = aots->dimensions[2];
    
    // Check that second arg has two dimensions
    if (aocor->nd != 2) {
        PyErr_SetString(ThreeWayError, "second arg ndarray must be NSxNC");
        return NULL;
    }
    
    // Check that number of subjects dimensions are equal
    if (aots->dimensions[0] != aocor->dimensions[0]) {
        PyErr_SetString(ThreeWayError, "number of subjects (first dimension of both arguments) must be the same");
        return NULL;
    }
    
    // Check correct size of NC 
    int nc = numEntries(nr);
    if (aocor->dimensions[1] != nc) {
        char buf[200];
        sprintf(buf, "second arg must be of size %dx%d for timeseries of size %dx%dx%d", ns, nc, ns, nr, nt);
        PyErr_SetString(ThreeWayError, buf);
        return NULL;
    }
    
    // Find the correlations
    // for (int n=0; n<ns; n++) {
        // int c = 0;
        // for (int i=0; i<nr; i++) {
            // for (int j=i; j<nr; j++) {
                // for (int k=j; k<nr; k++) {
                    // float cor = 0;
                    // for (int t=0; t<nt; t++) {
                        // float x1 = *(float *)PyArray_GETPTR3(aots, n, i, t);
                        // float x2 = *(float *)PyArray_GETPTR3(aots, n, j, t);
                        // float x3 = *(float *)PyArray_GETPTR3(aots, n, k, t);
                        // cor += x1*x2*x3;
                    // }
                    // float *ptr = (float *)PyArray_GETPTR2(aocor, n, c++);
                    // *ptr = cor;
                // }
            // }
        // }
    // }
	
	// Find distribution
	// int *hist = (int *)malloc(nr*4);
	// int *dist = (int *)malloc(nr*4);
	// if (hist == NULL || dist == NULL) {
		// PyErr_SetString(ThreeWayError, "Malloc failure");
		// return NULL;
	// }
	// calcDist(nr, hist, dist);
	
	// Dispatch threads
	// Try to ensure approximately equal work for all threads
	// std::thread ts[NCORE];
	// int step = (int)ceil(nc/NCORE);
	// int next = step;
	// int ti = 0;
	// int prevIdx = 0;
	// int prevC = 0;
	// for (int i=1; i<nroi; i++) {
		// if (dist[i] >= next) {
			// ts[ti++] = std::thread(calcCorr, prevIdx, i, prevC, nr, nt, aots, aocor);
			// prevIdx = i;
			// prevC = dist[i];
			// next += step;
		// }
	// }
	
	// Dispatch threads
	std::thread ts[NCORE];
	int step = (int)ceil(ns/NCORE);
	int ti = 0;
	for (int n=0; n<ns; n++) {
		if (n%step == 0) {
			ts[ti++] = std::thread(calcCorr, n, n+step, 
		}
	}
	
	// Join threads
	for (int i=0; i<NCORE; i++) {
		if (ts[i].joinable()) {
			ts[i].join();
		}
	}
    
	free(hist);
	free(dist);
	
    Py_INCREF(aocor);
    
    return (PyObject *)aocor;
}
    
static PyMethodDef ThreeWayMethods[] = {
    {"ThreeWay", ThreeWay_ThreeWay, METH_VARARGS, "Three way correlation coefficients of a normalized multi-subject roi timeseries data"},
    {NULL, NULL, NULL}
};

static struct PyModuleDef ThreeWaymodule = {
    PyModuleDef_HEAD_INIT,
    "ThreeWay",
    NULL,
    -1,
    ThreeWayMethods
};

PyMODINIT_FUNC
PyInit_ThreeWay(void)
{
    PyObject *m;
    
    // https://stackoverflow.com/questions/47026900/pyarray-check-gives-segmentation-fault-with-cython-c
    import_array();
    
    m = PyModule_Create(&ThreeWaymodule);
    if (m == NULL) {
        return NULL;
    }
    
    ThreeWayError = PyErr_NewException("ThreeWay.error", NULL, NULL);
    Py_XINCREF(ThreeWayError);
    if (PyModule_AddObject(m, "error", ThreeWayError) < 0) {
        Py_XDECREF(ThreeWayError);
        Py_CLEAR(ThreeWayError);
        Py_DECREF(m);
        return NULL;
    }
    
    return m;
}

