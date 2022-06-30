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


