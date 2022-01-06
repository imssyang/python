
WIDGET_NAME = None
WIDGET_X = 50
WIDGET_Y = 50

def init(name):
    global WIDGET_NAME
    WIDGET_NAME = name
    print(f"\ninit {WIDGET_NAME}")

def size():
    global WIDGET_X, WIDGET_Y
    print(f"size ({WIDGET_X}, {WIDGET_Y})")
    return WIDGET_X, WIDGET_Y

def resize(x, y):
    global WIDGET_X, WIDGET_Y
    WIDGET_X, WIDGET_Y = x, y
    print(f"resize ({WIDGET_X}, {WIDGET_Y})")

def dispose():
    global WIDGET_NAME, WIDGET_X, WIDGET_Y
    WIDGET_NAME = None
    WIDGET_X = WIDGET_Y = 0
    print(f"dispose")
