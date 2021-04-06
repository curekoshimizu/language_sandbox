from __future__ import annotations


class House:
    """The Product"""

    def __init__(self) -> None:
        # brick, wood, straw, ice
        self._wall_material = "Brick"
        # Apartment, Bungalow, Caravan, Hut, Castle, Duplex, HouseBoat, Igloo
        self._building_type = "Apartment"
        self._doors: int = 0
        self._windows: int = 0

    def set_wall_material(self, value: str) -> None:
        self._wall_material = value

    def set_building_type(self, value: str) -> None:
        self._building_type = value

    def set_number_doors(self, value: int) -> None:
        self._doors = value

    def set_number_windows(self, value: int) -> None:
        self._windows = value

    def __str__(self) -> str:
        return "This is a {0} {1} with {2} door(s) and {3} window(s).".format(
            self._wall_material, self._building_type, self._doors, self._windows
        )


class IglooDirector:
    """The Director, building a different representation."""

    @staticmethod
    def construct() -> House:
        house = House()
        house.set_building_type("Igloo")
        house.set_wall_material("Ice")
        house.set_number_doors(1)
        house.set_number_windows(0)
        return house


class HouseBoatDirector:
    """The Director, building a different representation."""

    @staticmethod
    def construct() -> House:
        house = House()
        house.set_building_type("House Boat")
        house.set_wall_material("Wooden")
        house.set_number_doors(6)
        house.set_number_windows(8)
        return house


class CastleDirector:
    """The Director, building a different representation."""

    @staticmethod
    def construct() -> House:
        house = House()
        house.set_building_type("Castle")
        house.set_wall_material("Granite")
        house.set_number_doors(100)
        house.set_number_windows(200)
        return house


def test_builder() -> None:
    print("[builder]")
    igloo = IglooDirector.construct()
    house_boat = HouseBoatDirector.construct()
    castle = CastleDirector.construct()
    print(igloo)
    print(house_boat)
    print(castle)
