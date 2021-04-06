from __future__ import annotations

import abc
from abc import abstractclassmethod
from typing import Callable, List


class Radio:

    """A radio.     It has a scan button, and an AM/FM toggle switch."""

    def __init__(self) -> None:
        """We have an AM state and an FM state"""
        self.amstate = AmState(self)
        self.fmstate = FmState(self)
        self.state: State = self.amstate

    def toggle_amfm(self) -> None:
        self.state.toggle_amfm()

    def scan(self) -> None:
        self.state.scan()


class State(abc.ABC):
    def __init__(self) -> None:
        self.pos = 0
        self._stations: List[str] = []
        self._name = "Unknown"

    def scan(self) -> None:
        """Scan the dial to the next station"""
        self.pos += 1
        if self.pos == len(self._stations):
            self.pos = 0
        print("Scanning... Station is {} {}".format(self._stations[self.pos], self._name))

    @abstractclassmethod
    def toggle_amfm(self) -> None:
        ...


class AmState(State):
    def __init__(self, radio: Radio) -> None:
        super().__init__()
        self.radio = radio
        self._stations = ["1250", "1380", "1510"]
        self._name = "AM"

    def toggle_amfm(self) -> None:
        print("Switching to FM")
        self.radio.state = self.radio.fmstate


class FmState(State):
    def __init__(self, radio: Radio) -> None:
        super().__init__()
        self.radio = radio
        self._stations = ["81.3", "89.1", "103.9"]
        self._name = "FM"

    def toggle_amfm(self) -> None:
        print("Switching to AM")
        self.radio.state = self.radio.amstate


def test_state() -> None:
    print("[state]")
    radio = Radio()
    actions: List[Callable[[], None]] = [radio.scan] * 2 + [radio.toggle_amfm] + [radio.scan] * 2
    actions *= 2
    for action in actions:
        action()
