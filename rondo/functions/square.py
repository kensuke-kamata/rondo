import rondo
from rondo.function import Function

class Square(Function):
    def forward(self, x):
        return x ** 2
