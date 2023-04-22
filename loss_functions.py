import tensorflow

def mean_squared_error(output, expected):
    return tensorflow.reduce_mean(tensorflow.square(expected - output))
