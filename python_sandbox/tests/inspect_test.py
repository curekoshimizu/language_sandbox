import functools
import inspect
import time
from typing import Any, Callable


class A:
    def hello(self, dt: float) -> str:
        time.sleep(dt)
        return "hello"


class B(A):
    def hello(self, dt: float) -> str:
        ret = super().hello(dt) + " world B!"
        time.sleep(0.02)
        return ret


class C(A):
    def hello(self, dt: float) -> str:
        return super().hello(dt) + " world C!"


def wrap_class_method(cls: Any) -> None:
    def decorate_func(class_name: str, f: Callable[..., Any]) -> Any:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            stime = time.time()
            ret = f(*args, **kwargs)
            print(f"{class_name}.{f.__name__} : elapsed time = {(time.time() - stime):.3f} sec")
            return ret

        return wrapper

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        print(name)
        setattr(cls, name, decorate_func(cls.__name__, method))


def test_class_method_hook() -> None:
    wrap_class_method(A)
    wrap_class_method(B)
    wrap_class_method(C)

    b = B()
    c = C()

    print(b.hello(0.01))
    print(c.hello(0.05))
