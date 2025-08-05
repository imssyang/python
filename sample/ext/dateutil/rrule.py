from dateutil.rrule import *
from dateutil.parser import *
from datetime import *

import pprint
import sys
sys.displayhook = pprint.pprint

r = rrule(DAILY, count=10, dtstart=parse("19970902T090000"))
print(r)
for d in list(r):
    print(d)

r2 = rrule(DAILY,
    dtstart=parse("19970902T090000"),
    until=parse("19970905T000000"))
print(r2)
for d in list(r2):
    print(d)

r3 = rrule(DAILY, interval=2, count=5,
    dtstart=parse("19970902T090000"))
print(r3)
for d in list(r3):
    print(d)

r4 = rrule(MONTHLY, interval=2, count=10,
    byweekday=(SU(1), SU(-1)),
    dtstart=parse("19970907T090000"))
print(r4)
for d in list(r4):
    print(d)

r5 = rrule(YEARLY, bymonth=1, byweekday=range(2),
    dtstart=parse("19980101T090000"),
    until=parse("19990110T090000"))
print(r5)
for d in list(r5):
    print(d)

r6 = rrule(WEEKLY, count=10, wkst=SU, byweekday=(TU,TH),
    dtstart=parse("19970902T090000"))
print(r6)
for d in list(r6):
    print(d)




