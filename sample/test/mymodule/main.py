import mymath.add              # a)导入模块X.add
import mymath.sub as sub       # b)导入模块X.sub并创建别名sub

from mymath.mul import mul     # c)从X.mul模块中导入函数mul
from mymath import dev         # d)从X包中导入模块X.dev

import mymath_init as m4       # e)导入包X并创建别名m4

def letscook(x, y, oper):
    if oper == "+":
        r = mymath.add.add(x, y)  # a) package.module.function
        r = m4.add(x, y)          # e) package.function
    elif oper == "-":
        r = sub.sub(x, y)         # b) module-alias.function
        r = m4.sub(x, y)          # e) package.function
    elif oper == "*":
        r = mul(x, y)             # c) function
        r = m4.mul(x, y)          # e) package.function
    else:
        r = dev.dev(x, y)         # d) module.function
        r = m4.dev(x, y)          # e) package.function
    print("{} {} {} = {}".format(x, oper, y, r))

x, y = 3, 8
letscook(x, y, "+")
letscook(x, y, "-")
letscook(x, y, "*")
letscook(x, y, "/")
