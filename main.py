import time
import tensorflow

def get_mean_squared_error(expected, actual):
    return tensorflow.reduce_mean(tensorflow.square(expected - actual))

class DenseLayer:
    def __init__(self, input_count):
        self.input_count = input_count

    def build(self, output_count):
        self.weights = tensorflow.Variable(tensorflow.constant(1., shape=(output_count, self.input_count,)), name="weights")
        self.bias = tensorflow.Variable(tensorflow.zeros((output_count,)), name="bias")

    def variables(self):
        return [self.weights, self.bias]

    def calculate(self, input):
        return tensorflow.linalg.matvec(self.weights, input) + self.bias

class Model:
    def __init__(self, layers, output_count):
        self.layers = layers
        for i in range(len(layers) - 1):
            layers[i].build(layers[i + 1].input_count)
        layers[-1].build(output_count)

    def variables(self):
        return [variable for layer in self.layers for variable in layer.variables()]
    
    def calculate(self, input):
        result = input
        for layer in self.layers:
            result = layer.calculate(result)
        return result
    
    def loss(self, input, expected_output):
        return get_mean_squared_error(expected_output, self.calculate(input))

@tensorflow.function
def combined_loss(model, input_values, expected_values):
    result = 0
    for (input, expected) in zip(tensorflow.unstack(input_values), tensorflow.unstack(expected_values)):
        result += model.loss(input, expected)
    return result

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = tensorflow.Variable(learning_rate)

    def __call__(self, variables, gradients):
        for (gradient, variable) in zip(gradients, variables):
            variable.assign_sub(self.learning_rate * gradient)

@tensorflow.function
def training_step(model, variables, inputs, expected_outputs, optimizer):
    with tensorflow.GradientTape() as tape:
        loss = combined_loss(model, inputs, expected_outputs)
        gradients = tape.gradient(loss, variables)
        optimizer(variables, gradients)
        return loss

model = Model([
    DenseLayer(2),
], 2)
variables = model.variables()
inputs = tensorflow.random.uniform((1000, 2), -2, 2)
expected_outputs = tensorflow.reverse(inputs, axis=[1])
initial_loss = combined_loss(model, inputs, expected_outputs)
print("Initial loss: %s" % initial_loss.numpy())

optimizer = GradientDescent(0.0001)
starting_time = time.time()
for i in range(5000):
    loss = training_step(model, variables, inputs, expected_outputs, optimizer)
    if i % 1000 == 0:
        print("%s: Loss: %s" % (i, loss.numpy()))

ending_time = time.time()
loss = combined_loss(model, inputs, expected_outputs)
print("Final loss: %s (in %s seconds)" % (loss.numpy(), ending_time - starting_time))
for variable in variables:
    print("Variable: %s, value: %s" % (variable.name, variable.numpy()))
