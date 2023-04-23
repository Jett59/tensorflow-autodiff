import time
import tensorflow

from loss_functions import mean_squared_error
from dense_layer import DenseLayer
from activation import relu

class Model:
    def __init__(self, layers, output_count, loss_function):
        self.layers = layers
        for i in range(len(layers) - 1):
            layers[i].build(layers[i + 1].input_count)
        layers[-1].build(output_count)
        self.loss_function = loss_function

    def variables(self):
        return [variable for layer in self.layers for variable in layer.variables()]
    
    def calculate(self, input):
        result = input
        for layer in self.layers:
            result = layer.calculate(result)
        return result
    
    def loss(self, input, expected_output):
        return self.loss_function(self.calculate(input), expected_output)

    @tensorflow.function
    def combined_loss(self, input_values, expected_values):
        # Run the model over all of the inputs and sum the losses.
        return tensorflow.reduce_sum(tensorflow.vectorized_map(lambda x: self.loss(x[0], x[1]), (input_values, expected_values)))

class GradientDescent:
    def __init__(self, learning_rate):
        self.learning_rate = tensorflow.Variable(learning_rate)

    def __call__(self, variables, gradients):
        for (gradient, variable) in zip(gradients, variables):
            variable.assign_sub(self.learning_rate * gradient)

@tensorflow.function
def training_step(model, variables, inputs, expected_outputs, optimizer):
    with tensorflow.GradientTape() as tape:
        loss = model.combined_loss(inputs, expected_outputs)
        gradients = tape.gradient(loss, variables)
        optimizer(variables, gradients)
        return loss

model = Model([
    DenseLayer(2, relu),
], 2, mean_squared_error)
variables = model.variables()
inputs = tensorflow.random.uniform((10000, 2), 0, 4)
expected_outputs = tensorflow.reverse(inputs, axis=[1])
batch_size = 32
evenly_divisible_batch_count = inputs.shape[0] // batch_size
batched_inputs = tensorflow.reshape(inputs[:evenly_divisible_batch_count * batch_size], (evenly_divisible_batch_count, batch_size, inputs.shape[1]))
batched_expected_outputs = tensorflow.reshape(expected_outputs[:evenly_divisible_batch_count * batch_size], (evenly_divisible_batch_count, batch_size, expected_outputs.shape[1]))

optimizer = GradientDescent(0.001)
starting_time = time.time()
for i in range(10):
    print("Epoch %s" % (i + 1))
    loss = 0
    for batch_index, (batched_input, batched_expected_output) in enumerate(zip(tensorflow.unstack(batched_inputs), tensorflow.unstack(batched_expected_outputs))):
        loss = training_step(model, variables, batched_input, batched_expected_output, optimizer)
        print("%s/%s: Loss: %s\t\t\t" % (batch_index + 1, len(batched_inputs), loss.numpy()), end="\r")
    print()

ending_time = time.time()
loss = model.combined_loss(inputs, expected_outputs)
print("Final loss: %s (in %s seconds)" % (loss.numpy(), ending_time - starting_time))
for variable in variables:
    print("Variable: %s, value: %s" % (variable.name, variable.numpy()))
tensorflow.keras.Model.fit