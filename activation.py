import tensorflow


def relu(x):
    return tensorflow.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + tensorflow.exp(-x))


def softmax(x):
    return tensorflow.exp(x) / tensorflow.reduce_sum(tensorflow.exp(x), axis=0)
