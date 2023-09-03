import rondo
import rondo.utils as U

class Model(rondo.Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return U.plot(y, verbose=True, to_file=to_file)
