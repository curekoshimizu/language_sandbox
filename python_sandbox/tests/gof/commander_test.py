import abc
from typing import List


class Command(abc.ABC):
    @abc.abstractclassmethod
    def execute(self, filename: str) -> None:
        ...

    @abc.abstractclassmethod
    def undo(self) -> None:
        ...


class HideFileCommand(Command):
    def __init__(self) -> None:
        # an array of files hidden, to undo them as needed
        self._hidden_files: List[str] = []

    def execute(self, filename: str) -> None:
        print(f"hiding {filename}")
        self._hidden_files.append(filename)

    def undo(self) -> None:
        filename = self._hidden_files.pop()
        print(f"un-hiding {filename}")


class DeleteFileCommand(Command):
    def __init__(self) -> None:
        # an array of deleted files, to undo them as needed
        self._deleted_files: List[str] = []

    def execute(self, filename: str) -> None:
        print(f"deleting {filename}")
        self._deleted_files.append(filename)

    def undo(self) -> None:
        filename = self._deleted_files.pop()
        print(f"restoring {filename}")


class MenuItem:
    """
    The invoker class. Here it is items in a menu.
    """

    def __init__(self, command: Command) -> None:
        self._command = command

    def on_do_press(self, filename: str) -> None:
        self._command.execute(filename)

    def on_undo_press(self) -> None:
        self._command.undo()


def test_commander() -> None:
    print("[commander]")
    item1 = MenuItem(DeleteFileCommand())
    item2 = MenuItem(HideFileCommand())
    test_file_name = "test-file"
    item1.on_do_press(test_file_name)
    item1.on_undo_press()
    item2.on_do_press(test_file_name)
    item2.on_undo_press()
