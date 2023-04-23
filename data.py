import tensorflow

def one_hot_encode_symbol(symbol, dictionary):
    return tensorflow.one_hot(dictionary.index(symbol), len(dictionary))

def one_hot_encode_sequence(sequence, dictionary):
    return  tensorflow.reshape(tensorflow.stack([one_hot_encode_symbol(symbol, dictionary) for symbol in sequence]), shape=(len(sequence) * len(dictionary),))

def pad_end(value, length):
    return tensorflow.pad(value, [[0, length - value.shape[0]]])
