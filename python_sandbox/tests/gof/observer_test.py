from __future__ import annotations

import abc
from abc import abstractmethod
from typing import Set


class ObserverInterface(abc.ABC):
    @abstractmethod
    def on_notify(self, observable: SubjectInterface, message: str) -> None:
        ...


class Observer(ObserverInterface):
    def __init__(self, prefix: str):
        self._prefix: str = prefix

    def on_notify(self, observable: SubjectInterface, message: str) -> None:
        print(self._prefix, "Observer received. ", message)


class SubjectInterface(abc.ABC):
    @abstractmethod
    def subscribe(self, observer: ObserverInterface) -> None:
        ...

    @abstractmethod
    def unsubscribe(self, observer: ObserverInterface) -> None:
        ...

    @abstractmethod
    def notify(self, message: str) -> None:
        ...


class Subject(SubjectInterface):
    def __init__(self) -> None:
        self._observers: Set[ObserverInterface] = set()

    def subscribe(self, observer: ObserverInterface) -> None:
        self._observers.add(observer)

    def unsubscribe(self, observer: ObserverInterface) -> None:
        self._observers.remove(observer)

    def notify(self, message: str) -> None:
        for observer in self._observers:
            observer.on_notify(self, message)


def test_observer() -> None:
    print("[observer]")
    subject = Subject()

    observer_a = Observer("I'am observer A. ")
    observer_b = Observer("I'am observer B. ")
    subject.subscribe(observer_a)
    subject.subscribe(observer_b)

    subject.notify("'Hello Observers 1'")

    print("detach observer_b")
    subject.unsubscribe(observer_b)
    subject.notify("'Hello Observers 2'")
    subject.unsubscribe(observer_a)
    subject.notify("'Hello Observers 3'")
