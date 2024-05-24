import time
import numpy as np

size = 1000000
a = np.random.rand(size)
b = np.random.rand(size)

# 计算点积
start_time = time.time()
c1 = np.dot(a, b)
np_dot_time = time.time() - start_time

# 计算点积
start_time = time.time()
c2 = 0
for i in range(size):
    c2 += a[i] * b[i]
manual_time = time.time() - start_time

# 输出执行时间
print(f"{c1} np.dot execution time: {np_dot_time:.6f} seconds")
print(f"{c2} For loop execution time: {manual_time:.6f} seconds")

