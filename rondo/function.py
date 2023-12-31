import weakref

import rondo
from rondo.config import Config
from rondo.variable import Variable

class Function:
    def __call__(self, *inputs):
        # Convert any input ndarrays to Variable instances to ensure consistent handling.
        inputs = [rondo.as_variable(input) for input in inputs]

        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        # Convert ys to Variable instnces and set their creators to this function.
        outputs = [Variable(rondo.as_array(y)) for y in ys]

        if Config.enable_backprop:
            # Calculate the generation for this function besed on its inputs.
            # This generation will be used to prioritize the function's backward computation.
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)

        self.inputs = inputs

        # Use weak references for the outputs to avoid circular references.
        self.outputs = [weakref.ref(output) for output in outputs]

        # If there's only one output, return it directly. Otherwise, return the tuple of outputs.
        return outputs if len(outputs) > 1 else outputs[0]

    def __lt__(self, other):
        # Compare just the memory addresses of the functions.
        # This ensures a unique order for functions with the same generation.
        return id(self) < id(other)

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()
