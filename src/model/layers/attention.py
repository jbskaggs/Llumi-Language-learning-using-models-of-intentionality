# from abc import ABC
#
import tensorflow as tf
from src.model.activations.activations import *
from abc import ABC
import numpy as np


class layer_normalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(layer_normalization, self).__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, input_tensor, **kwargs):
        # return tf.linalg.normalize(input_tensor)[0]
        return self.norm(input_tensor)


class positional_embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, word_buffer_len):
        # model hyper parameter variables
        super(positional_embedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_buffer_len = word_buffer_len
        # self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.positional = self.positional_encoding(word_buffer_len)

    def call(self, batch_size, **kwargs):
        return tf.tile(tf.reshape(self.positional, [1, self.word_buffer_len, self.embedding_size]), [batch_size, 1, 1])

    def positional_encoding(self, max_len):
        pos = tf.expand_dims(tf.range(0, max_len), axis=1)
        i = tf.expand_dims(tf.range(0, self.embedding_size), axis=0)

        output = self.angle(pos, i)

        mask = np.arange(self.embedding_size)
        sin_mask = (1 + mask) % 2
        cos_mask = mask % 2
        sin_mask = tf.convert_to_tensor(sin_mask, tf.float32) * tf.sin(output)
        cos_mask = tf.convert_to_tensor(cos_mask, tf.float32) * tf.cos(output)

        output = sin_mask + cos_mask
        output = tf.expand_dims(output, axis=0)
        return tf.cast(output, dtype=tf.float32)

    def angle(self, pos, i):
        return tf.cast(pos, dtype=tf.float32) / tf.pow(10000, tf.cast(2 * (i // 2) / self.embedding_size, dtype=tf.float32))


class scaled_dot_product_attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(scaled_dot_product_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, input_tensors, mask=None, **kwargs):
        v, k, q = input_tensors
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        output = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            output += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        output = tf.nn.softmax(output, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(output, v)  # (..., seq_len_q, depth_v)

        return output


class feed_forward(tf.keras.layers.Layer):
    def __init__(self, filters=None, kernel_size=1, activation='relu', use_bias=True, **kwargs):
        super(feed_forward, self).__init__(**kwargs)
        self.filters = filters
        # self.d1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, use_bias=use_bias,
        #                                  activation=activation)
        # self.d2 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, use_bias=use_bias,
        #                                  activation=activation)
        self.d1 = tf.keras.layers.Dense(units=filters, use_bias=use_bias, activation=activation)
        self.d2 = tf.keras.layers.Dense(units=filters, use_bias=use_bias)
        self.layer_norm = layer_normalization()

    def build(self, input_shape):
        pass

    def call(self, input_tensor, **kwargs):
        output_tensor = self.d1(input_tensor)
        output_tensor = self.d2(output_tensor)
        output_tensor += input_tensor
        return self.layer_norm(output_tensor)


class multi_head_attention(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads):
        super(multi_head_attention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.dense_value = tf.keras.layers.Dense(embedding_size)
        self.dense_key = tf.keras.layers.Dense(embedding_size)
        self.dense_query = tf.keras.layers.Dense(embedding_size)
        self.layer_norm = layer_normalization()

        self.depth = embedding_size // self.num_heads

        self.dot_product_attention = scaled_dot_product_attention()
        self.final_dense = tf.keras.layers.Dense(embedding_size)

    def build(self, input_shape):
        pass

    def split_heads(self, x, batch_size, depth):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, input_tensor, mask=None, **kwargs):
        v, k, q = input_tensor
        batch_size = tf.shape(q)[0]
        values = self.dense_value(v)
        keys = self.dense_key(k)
        queries = self.dense_query(q)

        values = self.split_heads(values, batch_size, self.depth)
        keys = self.split_heads(keys, batch_size, self.depth)
        queries = self.split_heads(queries, batch_size, self.depth)

        output = self.dot_product_attention([values, keys, queries], mask=mask)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embedding_size))
        output = self.final_dense(output) + q
        return self.layer_norm(output)


class attention_encoder(tf.keras.Model, ABC):
    def __init__(self, num_layers, embedding_len=1024, num_heads=2):
        super(attention_encoder, self).__init__(name='')
        self.num_layers = num_layers
        self.multi_head_attention_layers = []
        self.feed_forward_layers = []
        for i in range(num_layers):
            self.multi_head_attention_layers += [multi_head_attention(embedding_len, num_heads)]
            self.feed_forward_layers += [feed_forward(filters=embedding_len)]

    def call(self, input_tensor, mask=None, **kwargs):
        output_tensor = input_tensor
        output_layers = []
        for i in range(self.num_layers):
            output_tensor = self.multi_head_attention_layers[i]([output_tensor, output_tensor, output_tensor], mask)
            output_tensor = self.feed_forward_layers[i](output_tensor)
            output_layers += [output_tensor]
        return output_layers


class attention_decoder(tf.keras.Model, ABC):
    def __init__(self, num_layers, embedding_len=1024, num_heads=2):
        super(attention_decoder, self).__init__(name='')
        self.num_layers = num_layers
        self.multi_head_attention_layers = []
        self.multi_head_attention_layers_1 = []
        self.feed_forward_layers = []
        for i in range(num_layers):
            self.multi_head_attention_layers += [multi_head_attention(embedding_len, num_heads)]
            self.multi_head_attention_layers_1 += [multi_head_attention(embedding_len, num_heads)]
            self.feed_forward_layers += [feed_forward(filters=embedding_len)]

    def call(self, input_tensor, encoder_output=None, look_ahead_mask=None, padding_mask=None, **kwargs):
        output_tensor = input_tensor
        for i in range(self.num_layers):
            output_tensor = self.multi_head_attention_layers[i]([output_tensor, output_tensor, output_tensor], mask=look_ahead_mask)
            output_tensor = self.multi_head_attention_layers_1[i]([encoder_output[i], encoder_output[i], output_tensor], mask=padding_mask)
            output_tensor = self.feed_forward_layers[i](output_tensor)
        return output_tensor


