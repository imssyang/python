#include <stdio.h>
#include <Python.h>
#include <pyhelper.hpp>

int main()
{
    Py_SetPath(L"/opt/python3/lib/python3.8:/opt/python3/lib/python3.8/site-packages:/opt/python3/lib/python3.8/lib-dynload:/opt/python3/sample/custom/c2p/api/cpp");
	CPyInstance hInstance;

	CPyObject pName = PyUnicode_FromString("pyemb2");
	CPyObject pModule = PyImport_Import(pName);
	if(pModule) {
		CPyObject pFunc = PyObject_GetAttrString(pModule, "formatCommand2");
		if(pFunc && PyCallable_Check(pFunc)) {
			printf("C: call func!\n");
    		const char* str = "/sss/test -a 1 -b 2 -c 3 adsf";
    		PyObject* arg = Py_BuildValue("(s)", str);
			CPyObject pValue = PyObject_CallObject(pFunc, arg);
			printf("C: formatCommand() = %s\n", PyUnicode_AsUTF8(pValue));
		} else {
			printf("ERROR: function formatCommand: %p\n", pFunc);
		}
	} else {
		printf("ERROR: Module not imported\n");
	}

	return 0;
}
