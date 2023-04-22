import time
import tensorflow

def get_mean_squared_error(expected, actual):
    return tensorflow.reduce_mean(tensorflow.square(expected - actual))

class MyModel:
    def __init__(self, matrix_shape):
        self.matrix = tensorflow.Variable(tensorflow.constant(1., shape=matrix_shape), shape=matrix_shape, name="matrix")

    def variables(self):
        return [self.matrix]

    def calculate(self, input):
        return tensorflow.matmul(input, self.matrix)

    def loss(self, input, expected):
        return get_mean_squared_error(expected, self.calculate(input))

@tensorflow.function
def combined_loss(model, input_values, expected_values):
    result = 0
    for (input, expected) in zip(tensorflow.unstack(input_values), tensorflow.unstack(expected_values)):
        result += model.loss(input, expected)
    return result

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def __call__(self, variables, gradients):
        for (gradient, variable) in zip(gradients, variables):
            variable.assign_sub(gradient * self.learning_rate)

@tensorflow.function
def training_step(model, variables, inputs, expected_outputs, optimizer):
    with tensorflow.GradientTape() as tape:
        loss = combined_loss(model, inputs, expected_outputs)
        gradients = tape.gradient(loss, variables)
        optimizer(variables, gradients)
        return loss

model = MyModel((2, 2,))
variables = model.variables()
inputs = tensorflow.constant([[[2.0, 9.0]], [[20.0, 11.0]]])
expected_outputs = tensorflow.constant([[9.0, 2.0], [11.0, 20.0]])
initial_loss = combined_loss(model, inputs, expected_outputs)
print("Initial loss: %s" % initial_loss.numpy())

optimizer = GradientDescent(0.0025)
starting_time = time.time()
for i in range(50):
    loss = training_step(model, variables, inputs, expected_outputs, optimizer)

ending_time = time.time()
loss = combined_loss(model, inputs, expected_outputs)
print("Final loss: %s (in %s seconds)" % (loss.numpy(), ending_time - starting_time))
for variable in variables:
    print("Variable: %s, value: %s" % (variable.name, variable.numpy()))
