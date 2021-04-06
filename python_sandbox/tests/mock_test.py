from unittest import mock

import pytest

# NOTE: What's the difference with Mock and MagicMock?
# A. MagicMock derives Mock. So, use MagicMock!
# int(Mock()) -> Error, int(MagicMock()) : works
# MagicMock has attribute methods, for examples, addition, conversion and so on.


def test_mock_object() -> None:
    m1 = mock.MagicMock(a=1, b=2)
    assert m1.a == 1
    assert m1.b == 2
    assert m1.unknown_variable  # we can access!

    m2 = mock.MagicMock(**{"a.b.c.d": 100, "e.f": 200})  # we can make nested class variable
    assert m2.a.b.c.d == 100
    assert m2.e.f == 200


def test_mock_call() -> None:
    m1 = mock.MagicMock(a=1, b=2, return_value=3)

    assert m1() == 3
    m1.assert_called_once_with()

    assert m1(fuga="hello", piyo="world") == 3  # arguments will be ignored
    assert m1.call_count == 2

    m1.assert_called_with(fuga="hello", piyo="world")
    with pytest.raises(AssertionError):
        # because it was called two times.
        m1.assert_called_once_with(fuga="hello", piyo="world")

    m2 = mock.MagicMock(side_effect=[10, 11, 12])
    assert m2() == 10
    assert m2() == 11
    assert m2() == 12
    with pytest.raises(StopIteration):
        assert m2()

    m2.side_effect = [13, 14]
    assert m2() == 13
    assert m2() == 14
    with pytest.raises(StopIteration):
        assert m2()

    assert m2.call_count == 7


def test_mock_patch() -> None:
    from os import path

    with mock.patch("os.path.join", return_value="dummy"):
        assert path.join("a/b", "c/d") == "dummy"

    class Hoge:
        a = 10

        def __init__(self) -> None:
            self.b = 100

    # class scope
    assert Hoge.a == 10
    with mock.patch.object(Hoge, "a", 20):
        assert Hoge.a == 20

    # variable scope
    hoge1 = Hoge()
    assert hoge1.b == 100
    with mock.patch.object(hoge1, "b", 200):
        assert hoge1.b == 200


def test_mock_patch_user_class() -> None:
    from python_sandbox import user_class

    #
    # replace method. 2 ways.
    #

    # 1.
    with mock.patch("python_sandbox.user_class.User.get_age", return_value=30) as m:
        user = user_class.User("curekoshimizu", 18)
        assert user.name == "curekoshimizu"
        assert user.get_age() == 30
        assert m.call_count == 1
    assert user.get_age() == 18

    # 2.
    with mock.patch.object(user_class.User, "get_age", return_value=40) as m:
        user = user_class.User("curekoshimizu", 18)
        assert user.name == "curekoshimizu"
        assert user.get_age() == 40
        assert m.call_count == 1
    assert user.get_age() == 18

    #
    # replace member variable . 2 ways.
    #

    # 1.
    with mock.patch("python_sandbox.user_class.User.name", "dummy") as m:
        user = user_class.User("curekoshimizu", 18)
        assert user.name == "dummy"
        assert user.get_age() == 18

    # 2.
    with mock.patch.object(user_class.User, "name", "dummy") as m:
        user = user_class.User("curekoshimizu", 18)
        assert user.name == "dummy"
        assert user.get_age() == 18
