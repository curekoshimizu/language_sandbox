from typing import Callable, List

from typing_extensions import Protocol


class HasName(Protocol):
    @property
    def name(self) -> str:
        ...


class Dog:
    def __init__(self) -> None:
        self._name = "Dog"

    @property
    def name(self) -> str:
        return self._name

    def bark(self) -> str:
        return "woof!"

    def __repr__(self) -> str:
        return self.bark()


class Cat:
    def __init__(self) -> None:
        self._name = "Cat"

    @property
    def name(self) -> str:
        return self._name

    def meow(self) -> str:
        return "meow!"

    def __repr__(self) -> str:
        return self.meow()


class Human:
    def __init__(self) -> None:
        self._name = "Human"

    @property
    def name(self) -> str:
        return self._name

    def speak(self) -> str:
        return "'hello'"

    def __repr__(self) -> str:
        return self.speak()


class Car:
    def __init__(self) -> None:
        self._name = "Car"

    @property
    def name(self) -> str:
        return self._name

    def make_noise(self, octane_level: int) -> str:
        return "vroom{0}".format("!" * octane_level)


class Adapter:
    def __init__(self, obj: HasName, make_noise: Callable[[], str]) -> None:
        """We set the adapted methods in the object's dict"""
        self._obj: HasName = obj
        self._make_noise = make_noise

    @property
    def name(self) -> str:
        return self._obj.name

    def make_noise(self) -> str:
        return self._make_noise()


def test_adapter() -> None:
    print("[adapter]")
    objects: List[Adapter] = []
    dog = Dog()
    print(dog)
    objects.append(Adapter(dog, make_noise=dog.bark))
    cat = Cat()
    objects.append(Adapter(cat, make_noise=cat.meow))
    human = Human()
    objects.append(Adapter(human, make_noise=human.speak))
    car = Car()
    objects.append(Adapter(car, make_noise=lambda: car.make_noise(3)))
    for obj in objects:
        print("A {0} goes {1}".format(obj.name, obj.make_noise()))
