<<<<<<< HEAD
import sys

i = sys.maxsize
print(i)  # 9223372036854775807
print(i == i + 1)  # False
i += 1
print(i)  # 9223372036854775808

f = sys.float_info.max
print(f)  # 1.7976931348623157e+308
print(f == f + 1)  # True
f += 1
print(f)  # 1.7976931348623157e+308

i = 123
print(f"{i:05}")  # 00123

f = 123.456789
print(f"{f:.5}")  # 123.46
=======
class A:
    def __init__(self, **dict_data):
        self.__dict__ = dict_data

    def __getitem__(self, item):
        return getattr(self, item)

class B(A):
    def __init__(self):
        super().__init__(**{'b': 2, 'c': 3})
        print(self.b)    # 2
        print(self['c']) # 3

print(B())

import signal
print(int(signal.SIGKILL))

L = [1, 2]
S = [3, 4]
L += S
print(L)


from functools import wraps

def decorator_auto(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """A wrapper function"""
        func()
        print(self.__dict__) # {'a': 1}
    return wrapper

def decorator_manual(func):
    def wrapper(self, *args, **kwargs):
        """A wrapper function"""
        func()
        print(self.__dict__) # {'a': 1}
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

class Test:
    def __init__(self):
        self.a = 1

    @decorator_manual
    def doing():
        """A doing function"""
        print("doing function")

T = Test()
print(T.doing.__name__)  # doing
print(T.doing.__doc__)   # A doing function
print(T.doing())         # 123


>>>>>>> 9b71617f98d598638f7cf2cee366ed05c980c5b6
