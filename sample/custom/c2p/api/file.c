#include <stdio.h>
#include <Python.h>

int main()
{
	char filename[] = "pyemb.py";
	FILE* fp;

	Py_Initialize();

	fp = fopen(filename, "r");
	PyRun_SimpleFile(fp, filename);

	Py_Finalize();
	return 0;
}
