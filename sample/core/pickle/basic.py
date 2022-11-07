import pickle
import dill


class Example:
    a_number = 35
    a_string = "hey"
    a_list = [1, 2]
    a_dict = {"a": 1, "b": [2]}
    a_tuple = (22, 23)


e = Example()
e_pickled = pickle.dumps(e)  # Pickling
e_unpickled = pickle.loads(e_pickled)  # Unpickling
print(f"{pickle.HIGHEST_PROTOCOL} {pickle.DEFAULT_PROTOCOL}")  # 5 4
print(
    f"{e_pickled}"
)  # b'\x80\x04\x95\x1b\x00\x00\x00\x00\x00\x00\x00\x8c\x08__main__\x94\x8c\x07Example\x94\x93\x94)\x81\x94.'
print(f"{e_unpickled.a_dict}")  # {'a': 1, 'b': [2]}


s = lambda x: x * x
s_pickled = dill.dumps(s)
s_unpickled = dill.loads(s_pickled)
print(
    f"{s_pickled}"
)  # b'\x80\x04\x95\xbb\x00\x00\x00\x00\x00\x00\x00\x8c\ndill._dill\x94\x8c\x10_create_function\x94\x93\x94(h\x00\x8c\x0c_create_code\x94\x93\x94(K\x01K\x00K\x00K\x01K\x02KCC\x08|\x00|\x00\x14\x00S\x00\x94N\x85\x94)\x8c\x01x\x94\x85\x94\x8c(/opt/python3/sample/core/pickle/basic.py\x94\x8c\x08<lambda>\x94K\x14C\x00\x94))t\x94R\x94c__builtin__\n__main__\nh\nNN}\x94Nt\x94R\x94.'
print(f"{s_unpickled(3)}")  # 9


class Foobar:
    def __init__(self):
        self.a = 35
        self.b = "test"
        self.c = lambda x: x * x

    # To specify what you want to pickle
    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes["c"]
        return attributes

    # Do additional initializations while unpickling
    def __setstate__(self, state):
        self.__dict__ = state
        self.c = lambda x: x * x


f = Foobar()
f_pickled = pickle.dumps(f)  # filter by __getstate__()
f_unpickled = pickle.loads(f_pickled)  # filter by __setstate__()
print(
    f_unpickled.__dict__
)  # {'a': 35, 'b': 'test', 'c': <function Foobar.__setstate__.<locals>.<lambda> at 0x7fed7f771d30>}
