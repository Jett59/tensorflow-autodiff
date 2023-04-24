import numpy

def one_hot_encode_sequence(sequence, dictionary):
    result = numpy.array([0.] * len(dictionary) * len(sequence), dtype=numpy.float32)
    for i, letter in enumerate(sequence):
        result[i * len(dictionary) + dictionary.index(letter)] = 1
    return result

def pad_end(value, length):
    return numpy.pad(value, (0, length - len(value)), "constant")
