import tensorflow as tf
from src.activations.activations import *


class residual_layer(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters, activations, **kwargs):
        super(residual_layer, self).__init__(**kwargs)
        filters0, filters1, filters2 = filters
        activation0, activation1, activation2 = activations
        self.conv2a = tf.keras.layers.Conv2D(filters0, (1, 1), activation=activation0)
        self.conv2b = tf.keras.layers.Conv2D(filters1, kernel_size, padding='same', activation=activation1)
        self.conv2c = tf.keras.layers.Conv2D(filters2, (1, 1), activation=activation2)

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, input, **kwargs):
        x = self.conv2a(input)
        x = self.conv2b(x)
        x = self.conv2c(x)
        x += input
        return tf.nn.relu(x)


class residual_network(tf.keras.layers.Layer):
    def __init__(self, num_layers, kernel_sizes, filters, activations, **kwargs):
        super(residual_network, self).__init__(**kwargs)
        if len(kernel_sizes) != num_layers or len(filters) != num_layers or len(filters) != num_layers:
            raise Exception('invalid number of layers, kernels, or filters specified')
        self.layers = []
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers += [residual_layer(kernel_size=kernel_sizes[i], filters=filters[i], activation=activations[i])]

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, input, **kwargs):
        x = self.layers[0]()
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
        return x
