import time
from datetime import datetime

print(time.time())

st_time = time.strptime("2023-01-06 12:59:56", "%Y-%m-%d %H:%M:%S")
print(type(st_time))  # <class 'time.struct_time'>
raw_time = time.mktime(st_time)
print(raw_time, type(raw_time))  # 1656864840.0 <class 'float'>

dt = datetime.fromtimestamp(raw_time)
ss = dt.strftime("%Y-%m-%d %H:%M:%S")
print(ss)
