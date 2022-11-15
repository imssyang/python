# pylint: disable=undefined-variable
import traceback
import sys

try:
    do_stuff()
except Exception as err:
    """OUT
    (<class 'NameError'>, NameError("name 'do_stuff' is not defined"), <traceback object at 0x7f0b9424d540>)
    """
    print(sys.exc_info())

    """OUT
    Traceback (most recent call last):
        File "/opt/python3/sample/core/traceback/basic.py", line 5, in <module>
            do_stuff()
    NameError: name 'do_stuff' is not defined
    """
    print(traceback.format_exc())
    traceback.print_exception(*sys.exc_info())
    traceback.print_exc()

    """OUT
    File "/opt/python3/sample/core/traceback/basic.py", line 5, in <module>
        do_stuff()
    """
    traceback.print_tb(err.__traceback__)
