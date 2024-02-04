#include <pybind11/embed.h>
#include <iostream>

namespace py = pybind11;

int main() {
    py::scoped_interpreter guard{};

	py::module_ sys = py::module_::import("sys");
	py::print(sys.attr("path"));

	py::module_ calc = py::module_::import("calc");
	py::object result = calc.attr("add")(1, 2);
	int n = result.cast<int>();
	std::cout << n;
}
