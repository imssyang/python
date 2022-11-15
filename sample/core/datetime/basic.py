import time
from datetime import datetime

dt = datetime.fromtimestamp(time.time())
ss = dt.strftime("%Y-%m-%d %H:%M:%S")
print(ss)
