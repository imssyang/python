class Widget(object):
    def __init__(self, name):
        print("\n__init__")
        self.name = name
        self.x = self.y = 50

    def size(self):
        print("size")
        return self.x, self.y

    def resize(self, x, y):
        print("resize")
        self.x, self.y = x, y

    def dispose(self):
        print("dispose")
        self.name = None
        self.x = self.y = 0
