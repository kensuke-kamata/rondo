class Variable:
    def __init__(self, data) -> None:
        self.data = data


if __name__ == '__main__':
    import numpy as np
    from function import Square

    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(y.data)
