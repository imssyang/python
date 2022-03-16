
sss = '0.23k'
print(5/3, 5//3, float(sss[:-1]), '' in 'aaa')

import os

print(os.path.exists('/opt/python3/sample/test/test2.py'))

TIME_THRESHOLD = os.getenv("GAME_TIME_THRESHOLD", 7200)
print(TIME_THRESHOLD, type(TIME_THRESHOLD))

import time
ss = time.time()
print(ss, type(ss))

from datetime import datetime

now = datetime.now()
print(now, type(now))

print(isinstance(True, bool)) # True
print(isinstance(1, bool))    # False

import multiprocessing
pairs = [['a','b'],['c','d'],['e','f'],['g','h']]

def printPairs(pair):
    print('pair =', pair)

def parallel(function, pairs):
    cpu_no = multiprocessing.cpu_count()
    if len(pairs) < cpu_no:
        cpu_no = len(pairs)
    print(cpu_no)

    p = multiprocessing.Pool(cpu_no)
    res = p.map_async(function, pairs, chunksize=2)
    print('3, p = ', p)
    print(res.get())
    print('4')
    p.close()
    print('5')
    p.join()
    print('6')
    return

if __name__ == '__main__':
    parallel(printPairs, pairs)
