from typing import Callable

def memoize(func: Callable):
    """Memoize function"""
    cache = {}
    def wrapped(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapped
