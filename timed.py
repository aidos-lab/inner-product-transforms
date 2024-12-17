import timeit
import functools


def timeit_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = timeit.Timer(lambda: func(*args, **kwargs))
        execution_time = timer.timeit(number=1)
        print(f"Function {func.__name__!r} executed in {execution_time:.4f} seconds")
        return func(*args, **kwargs)

    return wrapper


@timeit_decorator
def example_function():
    return "-".join(map(str, range(100)))


example_function()
