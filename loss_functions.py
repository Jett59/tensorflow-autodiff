import tensorflow


def mean_squared_error(output, expected):
    return tensorflow.reduce_mean(tensorflow.square(expected - output))


def binary_cross_entropy(output, expected):
    return tensorflow.reduce_mean(
        -(
            expected * tensorflow.math.log(output + 1e-8)
            + (1 - expected) * tensorflow.math.log(1 - output + 1e-8)
        )
    )
