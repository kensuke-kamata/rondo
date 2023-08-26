import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using(name, value):
    pre = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, pre)

def no_grad():
    return using('enable_backprop', False)
