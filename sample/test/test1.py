class A:
    def __set_name__(self, owner, name):
        print(
            f"Descriptor instance assigned to attribute '{name}' of class '{owner.__name__}'"
        )
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


class B:
    def __init__(self):
        self.a = A()


b = B()
b.abc = 10
print(b.abc)
