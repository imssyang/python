from operator import itemgetter

print(itemgetter('a', 'c')({'a':1, 'b':2, 'c':3})) # (1, 3)

print(itemgetter(1, 3, 5)('ABCDEFG')) # ('B', 'D', 'F')
print(itemgetter('a', 'c')({'a':1, 'b':2, 'c':3})) # (1, 3)
print({'a':1}.get('b'))
for i in range(0):
    print(i)
print(callable(None))
#print(len(None))
print(f"{None}")
if "":
    print("====False")
else:
    print("====True")
print(bool(""))

import os

# create
d1 = {'a': 1, 2: 'b', 'c': [3, 4, 5]}
d2 = dict({'a': 1, 2: 'b'})      # {'a': 1, 2: 'b'}
d3 = dict([('a', 1), (2, 'b')])  # {'a': 1, 2: 'b'}
d4 = {k:v for k,v in d3.items()} # {'a': 1, 2: 'b'}

# access
print(d1['a'], d1[2], d1.get('d'))  # 1 b [3, 4, 5]

# append
d2['c'] = 3
d3.update({'c': 3})
print(d2, d3, sep=os.linesep)

# update
d2['a'] = 11
d3.update({'a': 11})
print(d2, d3, sep=os.linesep)

# remove
d1.clear(); print(d1)   # {}
del d2['a']; print(d2)  # {2: 'b', 'c': 3}
print(d3.pop('a'), d3)  # 11 {2: 'b', 'c': 3}

# iterate
for key in d2:
    print(key)
for value in d2.values():
    print(value)
for k, v in d2.items():
    print(k, ':', v)

# truth
print('a' not in d3) # True
print(2 in d3)       # True
