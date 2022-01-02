# 对外提供类型、变量和接口，对用户隐藏各个子模块的实现。
import mymath_init.add
import mymath_init.sub
import mymath_init.mul
import mymath_init.dev

add = mymath_init.add.add
sub = mymath_init.sub.sub
mul = mymath_init.mul.mul
dev = mymath_init.dev.dev
