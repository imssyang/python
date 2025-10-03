import isodate

duration_str = "PT1H"
td = isodate.parse_duration(duration_str)
print(td)              # 1:00:00
print(type(td))        # <class 'datetime.timedelta'>
