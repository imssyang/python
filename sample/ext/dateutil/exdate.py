from dateutil.rrule import rrule, rruleset, DAILY
from datetime import datetime

# 基础规则：每天一次，从 9/24 开始
base_rule = rrule(
    freq=DAILY,
    dtstart=datetime(2025, 9, 24, 10, 0),
    until=datetime(2025, 9, 30, 23, 59),
)

# rruleset 允许添加排除
rules = rruleset()
rules.rrule(base_rule)

# 排除某些日期
rules.exdate(datetime(2025, 9, 26, 10, 0))
rules.exdate(datetime(2025, 9, 28, 10, 0))

for dt in rules:
    print(dt)
