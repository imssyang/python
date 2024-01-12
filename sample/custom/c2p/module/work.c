#include <Python.h> 

/* Execute func(x, y) in the Python interpreter. The 
arguments and return result of the function must 
be Python floats */
double call_func(PyObject *func, double x, double y) 
{ 
	PyObject *args; 
	PyObject *kwargs; 
	PyObject *result = 0; 
	double retval; 
	
	// Make sure we own the GIL 
	PyGILState_STATE state = PyGILState_Ensure(); 
	
	// Verify that func is a proper callable 
	if (!PyCallable_Check(func)) { 
		fprintf(stderr, "call_func: expected a callable\n"); 
		goto fail; 
	} 

	// Step3 
	args = Py_BuildValue("(dd)", x, y); 
	kwargs = NULL; 
	
	// Step 4 
	result = PyObject_Call(func, args, kwargs); 
	Py_DECREF(args); 
	Py_XDECREF(kwargs); 
	
	// Step 5 
	if (PyErr_Occurred()) { 
		PyErr_Print(); 
		goto fail; 
	} 
	
	// Verify the result is a float object 
	if (!PyFloat_Check(result)) { 
		fprintf(stderr, "call_func: callable didn't return a float\n"); 
		goto fail; 
	} 
	
	// Step 6 
	retval = PyFloat_AsDouble(result); 
	Py_DECREF(result); 
	
	// Step 7 
	PyGILState_Release(state); 
	return retval; 
	fail: 
		Py_XDECREF(result); 
		PyGILState_Release(state); 
		abort(); 
} 

/* Extension function for testing the C-Python callback */
PyObject* py_call_func(PyObject *self, PyObject *args) 
{ 
	PyObject * func; 

	double x, y, result; 
	if (! PyArg_ParseTuple(args, "Odd", &func, &x, &y)) { 
		return NULL; 
	} 
	result = call_func(func, x, y); 
	return Py_BuildValue("d", result); 
} 
