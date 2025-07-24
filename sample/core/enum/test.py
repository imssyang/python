from enum import Enum


class Lable(Enum):
    NONE = -1
    CLEAR = 0
    BLUR = 1
    BLACK = 2

    def to_str(self) -> str:
        mapping = {
            Lable.NONE: "none",
            Lable.CLEAR: "clear",
            Lable.BLUR: "blur",
            Lable.BLACK: "black",
        }
        return mapping[self]


quality = Lable(1)
print(quality)
print(quality.name)
print(quality.name.lower())
print(quality.value)
