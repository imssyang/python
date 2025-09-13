from dateutil.rrule import *
from dateutil.parser import *
from datetime import *


#for item in list(rrule(DAILY, count=10, dtstart=parse("19970902T090000"))):
#    print(item)


#for item in list(rrulestr("FREQ=DAILY;INTERVAL=10;COUNT=5", dtstart=parse("19970902T090000"))):
#    print(item)


print("FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20250915T090500")
for item in list(rrulestr(
    "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;UNTIL=20250915T090500",
    dtstart=parse("20250915T090000"),
)):
    print(item)


print("FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;INTERVAL=2")
for item in list(rrulestr(
    "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;INTERVAL=2;COUNT=10",
    dtstart=parse("20250915T090000"),
)):
    print(item)


