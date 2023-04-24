import tensorflow
import layer_base

class DenseLayer(layer_base.Layer):
    def __init__(self, input_count, activation):
        super().__init__()
        self.input_count = input_count
        self.activation = activation

    def build(self, output_count):
        self.weights = tensorflow.Variable(tensorflow.constant(1., shape=(output_count, self.input_count,)), name="weights")
        self.bias = tensorflow.Variable(tensorflow.zeros((output_count,)), name="bias")

    def variables(self):
        return [self.weights, self.bias]

    def calculate(self, input):
        if self.activation is None:
            return tensorflow.linalg.matvec(self.weights, input) + self.bias
        else:
            return self.activation(tensorflow.linalg.matvec(self.weights, input) + self.bias)
