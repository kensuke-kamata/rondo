class Variable:
    def __init__(self, data) -> None:
        self.data = data


if __name__ == '__main__':
    import numpy as np

    from function import *
    A = Square()
    B = Exp()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    c = A(b)
    print(c.data)
