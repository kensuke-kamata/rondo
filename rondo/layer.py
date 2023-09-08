import os
import weakref

import numpy as np

from rondo.variable import Parameter

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs  = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def _flatten(self, params_dict, key_parent=None):
        for name in self._params:
            obj = self.__dict__[name]
            key = f'{key_parent}/{name}' if key_parent else name
            if isinstance(obj, Layer):
                obj._flatten(params_dict, key)
            else:
                params_dict[key] = obj

    def save(self, path):
        params_dict = {}
        self._flatten(params_dict)

        array = {}
        for key, param in params_dict.items():
            if param is not None:
                array[key] = param.data

        try:
            np.savez_compressed(path, **array)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise e

    def load(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten(params_dict)

        for key, param in params_dict.items():
            param.data = npz[key]
