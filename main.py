import tensorflow

from data import one_hot_encode_sequence, pad_end
from loss_functions import binary_cross_entropy
from dense_layer import DenseLayer
from activation import relu, sigmoid
from optimizer import StochasticGradientDescent
from model import Model

input_letter_count = 20
alphabet_length = 26
input_length = input_letter_count * alphabet_length
dictionary = "abcdefghijklmnopqrstuvwxyz"

def read_data():
    with open("spelling.txt", "r") as file:
        lines = file.readlines()
        words = [line.split(" ")[0] for line in lines]
        are_correctly_spelled = [line.split(" ")[1].strip() == "true" for line in lines]
        inputs = tensorflow.stack([pad_end(one_hot_encode_sequence(word, dictionary), input_length) for word in words])
        expected_outputs = tensorflow.stack([[1.] if is_correct else [0.] for is_correct in are_correctly_spelled])
        return inputs, expected_outputs


inputs, expected_outputs = read_data()
model = Model([
    DenseLayer(input_length, relu),
    DenseLayer(1, sigmoid)
], 1, binary_cross_entropy)
optimizer = StochasticGradientDescent(0.01)

model.train(inputs, expected_outputs, optimizer)

loss = model.combined_loss(inputs, expected_outputs)
print("Final loss: %s" % loss.numpy())
