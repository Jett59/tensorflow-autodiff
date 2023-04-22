import tensorflow

def get_mean_squared_error(expected, actual):
    return tensorflow.reduce_mean(tensorflow.square(expected - actual))

class MyModel:
    def __init__(self):
        self.matrix = tensorflow.Variable([[1., 1.], [1., 1.]])

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

model = MyModel()
variables = model.variables()
input_values = tensorflow.constant([[[8.0, 9.0]]])
expected_values = tensorflow.constant([[[10.0, 11.0]]])
initial_loss = combined_loss(model, input_values, expected_values)
print("Initial loss: %s" % initial_loss.numpy())

learning_rate = 0.01
for i in range(10):
    with tensorflow.GradientTape() as tape:
        loss = combined_loss(model, input_values, expected_values)
        gradients = tape.gradient(loss, variables)
        for (gradient, variable) in zip(gradients, variables):
            variable.assign_sub(gradient * learning_rate)
        loss = combined_loss(model, input_values, expected_values)
        print("Loss at step %d: %s" % (i, loss.numpy()))

loss = combined_loss(model, input_values, expected_values)
print("Final loss: %s" % loss.numpy())
