import tensorflow

def get_mean_squared_error(expected, actual):
    return tensorflow.reduce_mean(tensorflow.square(expected - actual))

class MyModel:
    def __init__(self, matrix):
        self.matrix = matrix
    
    def calculate(self, input):
        return tensorflow.matmul(input, self.matrix)
    
    @tensorflow.function
    def loss(self, input, expected):
        return get_mean_squared_error(expected, self.calculate(input))


matrix = tensorflow.Variable([[1.0, 2.0], [3.0, 4.0]])
model = MyModel(matrix)
input = tensorflow.constant([[8.0, 9.0]])
expected = tensorflow.constant([[10.0, 11.0]])
initial_loss = model.loss(input, expected)
print("Initial loss: %s" % initial_loss.numpy())

learning_rate = 0.01
for i in range(10):
    with tensorflow.GradientTape() as tape:
        loss = model.loss(input, expected)
        gradient = tape.gradient(loss, model.matrix)
        matrix.assign_sub(gradient * learning_rate)
        loss = model.loss(input, expected)
        print("Loss at step %d: %s" % (i, loss.numpy()))

loss = model.loss(input, expected)
print("Final loss: %s" % loss.numpy())
print("matrix: %s" % matrix.numpy())

