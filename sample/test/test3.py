import os

os.environ["VXCODE_DRY_RUN"] = "1"

print(os.getenv("VXCODE_POOL_SIZE"))
print(int(os.getenv("VXCODE_DRY_RUN")))

from datetime import datetime

now = datetime.now()  # current date and time
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
print(now_time)  # 2022-07-04 19:58:40

timestamp = 1528797322
tt_time = datetime.fromtimestamp(timestamp)
print(tt_time)  # 2018-06-12 17:55:22
at_time = tt_time.strftime("%m/%d/%Y %H:%M:%S")
print(at_time)  # 06/12/2018 17:55:22
