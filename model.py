import tensorflow
import time

class Model:
    def __init__(self, layers, output_count, loss_function):
        self.layers = layers
        for i in range(len(layers) - 1):
            layers[i].build(layers[i + 1].input_count)
        layers[-1].build(output_count)
        self.loss_function = loss_function

    def variables(self):
        return [variable for layer in self.layers for variable in layer.variables()]
    
    @tensorflow.function
    def calculate(self, input):
        result = input
        for layer in self.layers:
            result = layer.calculate(result)
        return result
    
    @tensorflow.function
    def loss(self, input, expected_output):
        return self.loss_function(self.calculate(input), expected_output)

    @tensorflow.function
    def combined_loss(self, input_values, expected_values):
        return tensorflow.reduce_sum(tensorflow.vectorized_map(lambda x: self.loss(x[0], x[1]), (input_values, expected_values)))

    @tensorflow.function
    def training_step(self, variables, inputs, expected_outputs, optimizer):
        with tensorflow.GradientTape() as tape:
            loss = self.combined_loss(inputs, expected_outputs)
            gradients = tape.gradient(loss, variables)
            optimizer(variables, gradients)
            return loss

    def train(self, inputs, expected_outputs, optimizer, batch_size=32, epochs=10):
        evenly_divisible_batch_count = inputs.shape[0] // batch_size
        batched_inputs = tensorflow.reshape(inputs[:evenly_divisible_batch_count * batch_size], (evenly_divisible_batch_count, batch_size, inputs.shape[1]))
        batched_expected_outputs = tensorflow.reshape(expected_outputs[:evenly_divisible_batch_count * batch_size], (evenly_divisible_batch_count, batch_size, expected_outputs.shape[1]))
        variables = self.variables()
        for i in range(epochs):
            print("Epoch %s" % (i + 1))
            start_time = time.time()
            loss = 0
            for batch_index, (batched_input, batched_expected_output) in enumerate(zip(tensorflow.unstack(batched_inputs), tensorflow.unstack(batched_expected_outputs))):
                loss = self.training_step(variables, batched_input, batched_expected_output, optimizer)
            end_time = time.time()
            print("Loss: %s (in %s seconds)" % (loss.numpy(), end_time - start_time))
