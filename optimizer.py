import tensorflow

class StochasticGradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = tensorflow.Variable(learning_rate)

    def __call__(self, variables, gradients):
        for (gradient, variable) in zip(gradients, variables):
            variable.assign_sub(self.learning_rate * gradient)
