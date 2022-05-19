import tensorflow as tf


def activation(x, activation):
    if activation is None:
        return x
    elif activation == 'relu':
        return tf.nn.relu(x)
    raise Exception("Not a valid activation")




