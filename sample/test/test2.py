from dataclasses import dataclass
from typing import Any


@dataclass
class Data:
    """Class for keeping track of an item in inventory."""

    a: str
    b: Any
    c: int = 5

    def calc(self) -> float:
        return self.b * self.c


print(Data("test", 3).calc())  # 15
print(Data("test", "3").calc())  # 33333
