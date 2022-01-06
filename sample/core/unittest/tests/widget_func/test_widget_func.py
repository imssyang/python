import unittest
from app import widget_func as Widget


def init():
    Widget.init('The widget')


def clean():
    Widget.dispose()


def default_widget_size():
    assert Widget.size() == (50, 50)


def widget_resize():
    Widget.resize(100, 150)
    assert Widget.size() == (100, 150)


test_func1 = unittest.FunctionTestCase(default_widget_size, setUp=init, tearDown=clean)
test_func2 = unittest.FunctionTestCase(widget_resize)

runner = unittest.TextTestRunner()
runner.run(test_func1)
runner.run(test_func2)
