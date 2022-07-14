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
