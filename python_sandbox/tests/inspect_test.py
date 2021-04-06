import inspect
import functools
import time


class A:
    def hello(self, dt: float) -> str:
        time.sleep(dt)
        return "hello"


class B(A):
    def hello(self) -> str:
        ret = super().hello(0.01) + " world B!"
        time.sleep(0.02)
        return ret


class C(A):
    def hello(self) -> str:
        return super().hello(0.05) + " world C!"



def wrap_class_method(cls):
    def decorate_func(class_name, f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            stime = time.time()
            ret = f(*args, **kwargs)
            print(f"{class_name}.{f.__name__} : elapsed time = {(time.time() - stime):.3f} sec")
            return ret

        return wrapper

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        print(name)
        setattr(cls, name, decorate_func(cls.__name__, method))


def test_class_method_hook():
    wrap_class_method(A)
    wrap_class_method(B)
    wrap_class_method(C)

    b = B()
    c = C()

    print(b.hello())
    print(c.hello())
