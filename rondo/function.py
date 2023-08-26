import rondo
from rondo.variable import Variable
from rondo.utils import as_array

class Function:
    def __call__(self, *inputs):
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()
