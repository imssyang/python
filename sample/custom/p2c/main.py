from ctypes import *

so = CDLL("./square.so")
print(type(so))
print(so.square(10))
print(so.square(8))
