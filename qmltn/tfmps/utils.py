import tensorflow as tf


def init_once(x, name):
    return tf.compat.v1.get_variable(name, initializer=x, trainable=False)
