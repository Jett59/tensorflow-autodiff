import tensorflow

from loss_functions import mean_squared_error
from dense_layer import DenseLayer
from activation import relu
from optimizer import StochasticGradientDescent
from model import Model

model = Model([
    DenseLayer(2, relu),
], 2, mean_squared_error)
optimizer = StochasticGradientDescent(0.001)
inputs = tensorflow.random.uniform((10000, 2), 0, 4)
expected_outputs = tensorflow.reverse(inputs, axis=[1])

model.train(inputs, expected_outputs, optimizer)

loss = model.combined_loss(inputs, expected_outputs)
print("Final loss: %s" % loss.numpy())
for variable in model.variables():
    print("Variable: %s, value: %s" % (variable.name, variable.numpy()))
