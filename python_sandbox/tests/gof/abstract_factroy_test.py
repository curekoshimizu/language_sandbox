import abc
from typing import Type


class Animal(abc.ABC):
    @abc.abstractclassmethod
    def speak(self) -> str:
        ...

    @abc.abstractclassmethod
    def __str__(self) -> str:
        ...


class Dog(Animal):
    def speak(self) -> str:
        return "woof"

    def __str__(self) -> str:
        return "Dog"


class Cat(Animal):
    def speak(self) -> str:
        return "meow"

    def __str__(self) -> str:
        return "Cat"


class PetShop:
    """A pet shop"""

    def __init__(self, animal: Type[Animal]) -> None:
        """pet_factory is our abstract factory.  We can set it at will."""

        self.pet_factory = animal

    def show_pet(self) -> None:
        """Creates and shows a pet using the abstract factory"""

        pet = self.pet_factory()
        print("We have a lovely {}".format(pet))
        print("It says {}".format(pet.speak()))


def test_abstract_factory() -> None:
    # A Shop that sells only cats
    print("[abstract factory]")
    cat_shop = PetShop(Cat)
    cat_shop.show_pet()
    dog_shop = PetShop(Dog)
    dog_shop.show_pet()
