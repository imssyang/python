#include <pyhelper.hpp>

int main()
{
	CPyInstance pInstance;

	CPyObject p;
	p = PyLong_FromLong(50);
	printf("Value = %ld\n", PyLong_AsLong(p));

	return 0;
}
