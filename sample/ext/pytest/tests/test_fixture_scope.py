# pytest -s test_fixture_scope.py --setup-show
# scope参数可以是session， module，class，function； 默认为function
# 1.session 会话级别（通常这个级别会结合conftest.py文件使用，所以后面说到conftest.py文件的时候再说）
# 2.module 模块级别： 模块里所有的用例执行前执行一次module级别的fixture
# 3.class 类级别 ：每个类执行前都会执行一次class级别的fixture
# 4.function ：前面实例已经说了，这个默认是默认的模式，函数级别的，每个测试用例执行前都会执行一次function级别的fixture
# 整个模块只执行了一次module级别的fixture ， 每个类分别执行了一次class级别的fixture， 而每一个函数之前都执行了一次function级别的fixture。
import pytest


@pytest.fixture(scope="module", autouse=True)
def module_fixture():
    print("\n-----------------")
    print("我是module fixture")
    print("-----------------")


@pytest.fixture(scope="class")
def class_fixture():
    print("\n-----------------")
    print("我是class fixture")
    print("-------------------")


@pytest.fixture(scope="function", autouse=True)
def func_fixture():
    print("\n-----------------")
    print("我是function fixture")
    print("-------------------")


def test_1():
    print("\n 我是test1")


@pytest.mark.usefixtures("class_fixture")
class TestFixture1(object):
    def test_2(self):
        print("\n我是class1里面的test2")

    def test_3(self):
        print("\n我是class1里面的test3")


@pytest.mark.usefixtures("class_fixture")
class TestFixture2(object):
    def test_4(self):
        print("\n我是class2里面的test4")

    def test_5(self):
        print("\n我是class2里面的test5")


if __name__ == "__main__":
    pytest.main(["-v", "--setup-show", "test_fixture.py"])
