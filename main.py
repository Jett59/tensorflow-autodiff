import tensorflow
import time

from data import one_hot_encode_sequence, pad_end
from loss_functions import categorical_cross_entropy
from dense_layer import DenseLayer
from activation import relu, softmax
from optimizer import StochasticGradientDescent
from model import Model

input_letter_count = 20
alphabet_length = 26
input_length = input_letter_count * alphabet_length
dictionary = "abcdefghijklmnopqrstuvwxyz"


def read_data():
    with open("spelling.txt", "r") as file:
        lines = file.readlines()
        split_lines = [line.split(" ") for line in lines]
        words = [line[0] for line in split_lines]
        are_correctly_spelled = [line[1] == "true" for line in split_lines]
        inputs = tensorflow.stack(
            [
                pad_end(one_hot_encode_sequence(word, dictionary), input_length)
                for word in words
            ]
        )
        expected_outputs = tensorflow.stack(
            [
                [1.0, 0.0] if is_correct else [0.0, 1.0]
                for is_correct in are_correctly_spelled
            ]
        )
        return inputs, expected_outputs


start_time = time.time()
inputs, expected_outputs = read_data()
print("Read data in %s seconds" % (time.time() - start_time))
model = Model(
    [
        DenseLayer(input_length, None),
        DenseLayer(2, softmax),
    ],
    2,
    categorical_cross_entropy,
)
optimizer = StochasticGradientDescent(0.1)

model.train(inputs, expected_outputs, optimizer, epochs=1)

loss = model.combined_loss(inputs, expected_outputs)
print("Final loss: %s" % loss.numpy())
