import abc
from dataclasses import dataclass
from typing import Type


@dataclass
class Dimension:
    width: int
    height: int
    depth: int


class Chair(abc.ABC):
    @abc.abstractclassmethod
    def dimension(self) -> Dimension:
        ...


class BigChair(Chair):
    def __init__(self) -> None:
        self._dim = Dimension(80, 80, 80)

    def dimension(self) -> Dimension:
        return self._dim


class MediumChair(Chair):
    def __init__(self) -> None:
        self._dim = Dimension(60, 60, 60)

    def dimension(self) -> Dimension:
        return self._dim


class SmallChair(Chair):
    def __init__(self) -> None:
        self._dim = Dimension(40, 40, 40)

    def dimension(self) -> Dimension:
        return self._dim


class ChairFactory:
    @staticmethod
    def get_chair(chair: Type[Chair]) -> Chair:
        return chair()


def test_factory() -> None:
    print("[factory]")
    chair_factory = ChairFactory.get_chair(SmallChair)
    print(chair_factory.dimension())
