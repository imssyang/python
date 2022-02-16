
def desc(status):
    match status:
        case 1:
            return "a"
        case 2:
            return "b"
        case 3 | 4 | 5:
            return "c"
        case _:
            return "x"

print(desc(5), desc(10))  # c x


def point(*value):
    match value: # value is an (x, y) tuple
        case (0, 0):
            print("Origin")
        case (0, y):
            print(f"Y={y}")
        case (x, 0):
            print(f"X={x}")
        case (x, y):
            print(f"X={x}, Y={y}")
        case _:
            raise ValueError("Not a point")

point(0, 2)  # X=1, Y=3
print(1, 0)
point(1, 2)  # X=1, Y=3

"""
class Point:
    x: int
    y: int

def point(value):
    match value:
        case Point(x=0, y=0):
            print("Origin")
        case Point(x=0, y=y):
            print(f"Y={y}")
        case Point(x=x, y=0):
            print(f"X={x}")
        case Point():
            print("Somewhere else")
        case _:
            print("Not a point")


Point(1, var)
Point(1, y=var)
Point(x=1, y=var)
Point(y=var, x=1)


match points:
    case []:
        print("No points")
    case [Point(0, 0)]:
        print("The origin")
    case [Point(x, y)]:
        print(f"Single point {x}, {y}")
    case [Point(0, y1), Point(0, y2)]:
        print(f"Two on the Y axis at {y1}, {y2}")
    case _:
        print("Something else")
"""



D = {1: 'a', 2: 'b'}
E = dict(D); E[1] = 'aa'
print(D, E)

killno = 4
print([i for i in range(1, killno-2+1)])

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
