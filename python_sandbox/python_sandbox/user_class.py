# User class is defined to test mock
class User:
    def __init__(self, name: str, age: int) -> None:
        self._name = name
        self._age = age

    @property
    def name(self) -> str:
        return self._name

    def get_age(self) -> int:
        return self._age
