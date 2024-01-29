#include <stdio.h>
#include <pyhelper.hpp>

int main()
{
    CPyInstance pyInstance;
	PyRun_SimpleString("print('Hello World from Embedded Python!!!')");

	return 0;
}
