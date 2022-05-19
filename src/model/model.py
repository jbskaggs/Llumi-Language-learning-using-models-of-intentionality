from src.model.layers.attention import attention_encoder, attention_decoder, positional_embedding, multi_head_attention, \
    feed_forward, layer_normalization
import tensorflow as tf
from abc import ABC
import numpy as np
import sys


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones(tf.stack([size, size])), -1, 0)
    # mask = tf.linalg.band_part(tf.ones((size, size)), 0, -1)
    return mask


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class autoencoder(tf.keras.Model, ABC):
    # @tf.function
    def __init__(self, batch_size, vocab_size, embedding_size, word_buffer_len, num_encoder_layers=3, num_heads=12):
        super(autoencoder, self).__init__(name='')
        self.batch_size = batch_size
        self.vocab_size = tf.cast(vocab_size, tf.int32)
        self.embedding_size = tf.cast(embedding_size, tf.int32)
        self.word_buffer_len = word_buffer_len
        self.num_layers = num_encoder_layers
        self.num_heads = num_heads
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, dtype=tf.float32)
        self.positional_embedding_layer = positional_embedding(self.vocab_size, self.embedding_size, word_buffer_len)
        self.encoder = attention_encoder(num_encoder_layers, self.embedding_size, num_heads)
        self.decoder = attention_decoder(num_encoder_layers, self.embedding_size, num_heads)
        self.i = 0
        self.embedding_size = tf.cast(self.embedding_size, tf.float32)

    @tf.function
    def call(self, input_tensor, target_tensor, mask, **kwargs):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_tensor, target_tensor)
        embedding = self.embedding_layer(tf.cast(input_tensor, tf.int32)) * tf.math.sqrt(
            self.embedding_size)

        # embedding, embed_norm = tf.linalg.normalize(embedding, axis=-1)
        # embedding *= self.embedding_size

        embedding += self.positional_embedding_layer(self.batch_size)
        encoder_output = self.encoder(embedding, enc_padding_mask)

        encoder_norm = tf.zeros((self.batch_size, 1))
        # for i in range(len(encoder_output)):
        #     encoder_output[i] -= self.positional_embedding_layer(self.batch_size)
        #     encoder_output[i], norm = tf.linalg.normalize(encoder_output[i])
        #     encoder_output[i] *= self.embedding_size * self.word_buffer_len
        #     encoder_output[i] += self.positional_embedding_layer(self.batch_size)
        #     encoder_norm += (tf.math.reduce_sum(norm) - tf.sqrt(self.embedding_size))**2

        dec_embedding = self.embedding_layer(tf.cast(target_tensor, tf.int32)) * tf.math.sqrt(
            self.embedding_size)
        # dec_embedding, dec_embed_norm = tf.linalg.normalize(dec_embedding, axis=-1)
        # dec_embedding *= self.embedding_size
        dec_embedding += self.positional_embedding_layer(self.batch_size)
        output_tensor = self.decoder(dec_embedding, encoder_output=encoder_output, look_ahead_mask=combined_mask,
                                     padding_mask=dec_padding_mask)
        output_tensor -= self.positional_embedding_layer(self.batch_size)

        # output_tensor, out_norm = tf.linalg.normalize(output_tensor, axis=-1)
        # output_tensor *= self.embedding_size

        vocab = self.embedding_layer(tf.range(self.vocab_size))
        # vocab, vocab_norm = tf.linalg.normalize(vocab, axis=-1)
        # vocab *= self.embedding_size
        output_tensor = tf.matmul(vocab, tf.transpose(output_tensor, perm=[0, 2, 1]))
        output_tensor = tf.transpose(output_tensor, perm=[0, 2, 1])
        token_ids = tf.argmax(output_tensor, axis=2)

        # vocab_norm_error = tf.math.reduce_sum(tf.abs(vocab_norm - tf.ones_like(vocab_norm))) / tf.cast(self.vocab_size, tf.float32)
        # out_norm_error = tf.math.reduce_sum(tf.abs(out_norm - tf.ones_like(out_norm)))
        return output_tensor, token_ids, encoder_norm
        # return output_tensor, token_ids, tf.zeros((self.batch_size, 1))


class skaggs_model(tf.keras.Model, ABC):
    # @tf.function
    def __init__(self, autoencoder, batch_size, word_buffer_len):
        super(skaggs_model, self).__init__(name='')
        # copy weights from the autoencoder
        self.word_buffer_len = word_buffer_len
        self.batch_size = batch_size
        self.vocab_size = autoencoder.vocab_size
        self.num_heads = autoencoder.num_heads
        self.embedding_size = tf.cast(autoencoder.embedding_size, tf.int32)
        self.num_layers = autoencoder.num_layers
        self.embedding_layer = autoencoder.embedding_layer
        self.positional_embedding_layer = autoencoder.positional_embedding_layer
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder

        # self.q0_transformation = []
        self.q0_multi_head_attention, self.q0_multi_head_attention2, self.q0_feed_forward = [], [], []
        self.q1_multi_head_attention, self.q1_multi_head_attention2, self.q1_feed_forward = [], [], []
        # self.st_multi_head_attention, self.st_feed_forward = [], []
        # self.st_transformation = []
        self.q_flatten = tf.keras.layers.Flatten()

        self.q0_one = tf.ones([self.batch_size, 1])
        self.q0_dense = tf.keras.layers.Dense(self.word_buffer_len * self.embedding_size)
        self.q1_one = tf.ones([self.batch_size, 1])
        self.q1_dense = tf.keras.layers.Dense(self.word_buffer_len * self.embedding_size)
        # self.st_one = tf.ones([self.batch_size, 1])
        # self.st_dense = tf.keras.layers.Dense(self.word_buffer_len * self.embedding_size)
        self.layer_norm = []
        for _ in range(self.num_layers):
            self.q0_multi_head_attention += [multi_head_attention(self.embedding_size, self.num_heads)]
            self.q0_multi_head_attention2 += [multi_head_attention(self.embedding_size, self.num_heads)]
            self.q0_feed_forward += [feed_forward(filters=self.embedding_size)]

            self.q1_multi_head_attention += [multi_head_attention(self.embedding_size, self.num_heads)]
            self.q1_multi_head_attention2 += [multi_head_attention(self.embedding_size, self.num_heads)]
            self.q1_feed_forward += [feed_forward(filters=self.embedding_size)]

            # self.st_multi_head_attention += [multi_head_attention(self.embedding_size, self.num_heads)]
            # self.st_feed_forward += [feed_forward(filters=self.embedding_size)]

            self.layer_norm += [layer_normalization()]

    def speech_var(self, true_false=False):
        self.embedding_layer.trainable = true_false
        self.positional_embedding_layer.trainable = true_false
        self.encoder.trainable = true_false
        self.decoder.trainable = true_false

    @tf.function
    def call(self, inputs, **kwargs):
        q0, q1, state, tar = inputs

        # q0 encoder
        q0_enc_padding_mask, combined_mask, q0_dec_padding_mask = create_masks(q0, tar)
        q0_embedding = self.embedding_layer(tf.cast(q0, tf.int32)) * tf.math.sqrt(
            tf.cast(self.embedding_size, tf.float32))
        q0_embedding += self.positional_embedding_layer(self.batch_size)
        q0_encoder_output = self.encoder(q0_embedding, q0_enc_padding_mask)

        # q1 encoder
        q1_enc_padding_mask, combined_mask, q1_dec_padding_mask = create_masks(q1, tar)
        q1_embedding = self.embedding_layer(tf.cast(q1, tf.int32)) * tf.math.sqrt(
            tf.cast(self.embedding_size, tf.float32))
        q1_embedding += self.positional_embedding_layer(self.batch_size)
        q1_encoder_output = self.encoder(q1_embedding, q1_enc_padding_mask)

        # set up tar embedding for decoder
        tar_embedding = self.embedding_layer(tf.cast(tar, tf.int32)) * tf.math.sqrt(
            tf.cast(self.embedding_size, tf.float32))
        tar_embedding += self.positional_embedding_layer(self.batch_size)

        # state transformation
        transformations = []
        q0_transformation = tf.reshape(self.q0_dense(self.q0_one), [self.batch_size, self.word_buffer_len, self.embedding_size])
        q1_transformation = tf.reshape(self.q1_dense(self.q1_one), [self.batch_size, self.word_buffer_len, self.embedding_size])
        # st_transformation = tf.reshape(self.st_dense(self.st_one), [self.batch_size, self.word_buffer_len, self.embedding_size])
        for i in range(self.num_layers):
            q0_transformation = self.q0_multi_head_attention[i]([q0_transformation, q0_transformation, q0_transformation])
            q0_transformation = self.q0_multi_head_attention[i]([q0_encoder_output[i], q0_encoder_output[i], q0_transformation])
            q0_transformation = self.q0_feed_forward[i](q0_transformation)

            q1_transformation = self.q1_multi_head_attention[i]([q1_transformation, q1_transformation, q1_transformation])
            q1_transformation = self.q1_multi_head_attention[i]([q1_encoder_output[i], q1_encoder_output[i], q1_transformation])
            q1_transformation = self.q1_feed_forward[i](q1_transformation)

            # st_transformation = self.st_multi_head_attention[i]([state, state, st_transformation])
            # st_transformation = self.st_feed_forward[i](st_transformation)

            transformations += [self.layer_norm[i](q0_transformation + q1_transformation)]  # + st_transformation)]

        # decoder
        output_tensor = self.decoder(tar_embedding, encoder_output=transformations, look_ahead_mask=combined_mask,
                                     padding_mask=q1_dec_padding_mask)

        output_tensor -= self.positional_embedding_layer(self.batch_size)

        vocab = self.embedding_layer(tf.range(self.vocab_size))
        output_tensor = tf.matmul(vocab, tf.transpose(output_tensor, perm=[0, 2, 1]))
        output_tensor = tf.transpose(output_tensor, perm=[0, 2, 1])
        token_ids = tf.argmax(output_tensor, axis=2)

        return output_tensor, token_ids


class state_speech(tf.keras.Model, ABC):
    # @tf.function
    def __init__(self, autoencoder, batch_size, word_buffer_len):
        super(state_speech, self).__init__(name='')
        # copy weights from the autoencoder
        self.word_buffer_len = word_buffer_len
        self.batch_size = batch_size
        self.vocab_size = autoencoder.vocab_size
        self.num_heads = autoencoder.num_heads
        self.embedding_size = tf.cast(autoencoder.embedding_size, tf.int32)
        self.num_layers = autoencoder.num_layers
        self.st_num_layers = 3
        # self.state_size = 326
        self.state_size = 48
        # self.state_size = 40  # I removed proposals because they don't seem to do anything
        self.embedding_layer = autoencoder.embedding_layer
        # self.embedding_layer.trainable = False
        self.positional_embedding_layer = autoencoder.positional_embedding_layer
        # self.positional_embedding_layer.trainable = False
        self.encoder = autoencoder.encoder
        # self.encoder.trainable = False
        self.layer_norms = []
        # for i in range(self.num_layers):
        #     self.layer_norms += [autoencoder.encoder.feed_forward_layers[i].layer_norm]
        self.decoder = autoencoder.decoder
        # self.decoder.trainable = False

        self.st_dense, self.st_multi_head_attention, self.st_multi_head_attention1, self.st_feed_forward = [], [], [], []
        self.st_layer_norm, self.st_transformation = [], []
        self.q_flatten = tf.keras.layers.Flatten()

        self.st_one = tf.ones([self.batch_size, 1])
        self.st_dense0 = tf.keras.layers.Dense(self.embedding_size)
        # self.st_dense1 = tf.keras.layers.Dense(self.embedding_size)
        self.st_dense2 = tf.keras.layers.Dense(self.embedding_size * self.word_buffer_len)
        # self.st_multi_head_attention0 = multi_head_attention(self.embedding_size, self.num_heads)
        # self.st_feed_forward0 = feed_forward(filters=self.embedding_size)
        for i in range(self.st_num_layers):
            # if i == 0:
            #     self.st_dense += [tf.keras.layers.Dense(self.state_size * self.word_buffer_len)]
            # else:
            self.st_dense += [tf.keras.layers.Dense(self.state_size)]
            self.st_layer_norm += [layer_normalization()]
            self.st_multi_head_attention += [multi_head_attention(self.embedding_size, self.num_heads)]
            self.st_multi_head_attention1 += [multi_head_attention(self.embedding_size, self.num_heads)]
            self.st_feed_forward += [feed_forward(filters=self.embedding_size)]
        self.loss = tf.keras.losses.MeanAbsoluteError()

    @tf.function
    def call(self, inputs, training=True, **kwargs):
        state, tar = inputs
        tar_enc_padding_mask, combined_mask, tar_dec_padding_mask = create_masks(tar, tar)

        # set up tar embedding for decoder
        tar_embedding = self.embedding_layer(tf.cast(tar, tf.int32)) * tf.math.sqrt(
            tf.cast(self.embedding_size, tf.float32))
        tar_embedding += self.positional_embedding_layer(self.batch_size)

        st_encoder_output = []

        new_state = tf.transpose(tf.reshape(tf.repeat(state, self.word_buffer_len, axis=1),
                                            [self.batch_size, self.state_size, self.word_buffer_len]), perm=[0, 2, 1])
        for i in range(self.st_num_layers):
            if i == 0:
                st_encoder_output += [self.st_layer_norm[i](self.st_dense[i](new_state))]
            else:
                st_encoder_output += [self.st_layer_norm[i](st_encoder_output[-1] + self.st_dense[i](st_encoder_output[-1]))]
        st_intro = tf.reshape(self.st_dense2(state), (self.batch_size, self.word_buffer_len, self.embedding_size))
        st_intro = self.st_dense0(st_intro)
        # # st_intro = tf.reshape(self.st_dense1(st_intro), (self.batch_size, self.embedding_size))
        # st_intro = tf.reshape(self.st_dense2(st_intro), (self.batch_size, self.word_buffer_len, self.embedding_size))
        # st_intro = self.st_multi_head_attention0([st_intro, st_intro, st_intro], mask=create_look_ahead_mask(tf.shape(tar)[1]))
        # st_intro = self.st_feed_forward0(st_intro)

        # state transformation
        # st_transformation = self.positional_embedding_layer(self.batch_size) + \
        #                     tf.stack(self.st_dense0(self.st_one), [self.batch_size, self.embedding_size], axis=2)
        # tf.transpose(tf.reshape(tf.repeat(self.st_dense0(self.st_one), self.word_buffer_len, axis=1),
        #                         [self.batch_size, self.embedding_size, self.word_buffer_len]), perm=[0, 2, 1])
        transformations = []
        st_transformation = tar_embedding + st_intro
        for i in range(self.st_num_layers):
            st_transformation = self.st_multi_head_attention1[i](
                [st_encoder_output[i], st_encoder_output[i], st_transformation])
            st_transformation = self.st_multi_head_attention[i](
                [st_transformation, st_transformation, st_transformation], combined_mask)
            st_transformation = self.st_feed_forward[i](st_transformation)
            transformations += [st_transformation]

        # ##### tar encoder #####
        # tar_embedding = self.embedding_layer(tf.cast(tar, tf.int32)) * tf.math.sqrt(
        #     tf.cast(self.embedding_size, tf.float32))
        # tar_embedding += self.positional_embedding_layer(self.batch_size)
        # tar_encoder_output = self.encoder(tar_embedding, tar_enc_padding_mask)

        # sum_squared_error = 0
        # for i in range(self.num_layers):
        #     error = tar_encoder_output[i] - transformations[i]
        #     sum_squared_error += tf.math.reduce_sum(tf.square(error))
        # mse = self.loss(tf.stack(tar_encoder_output, axis=3), tf.stack(transformations, axis=3))

        # ##### tar decoder #####
        # output_tensor = self.decoder(tar_embedding, encoder_output=transformations, look_ahead_mask=combined_mask,
        #                              padding_mask=tar_dec_padding_mask)
        # output_tensor -= self.positional_embedding_layer(self.batch_size)

        vocab = self.embedding_layer(tf.range(self.vocab_size))
        output_tensor = tf.matmul(vocab, tf.transpose(transformations[-1], perm=[0, 2, 1]))
        output_tensor = tf.transpose(output_tensor, perm=[0, 2, 1])
        token_ids = tf.argmax(output_tensor, axis=2)

        if training:
            return output_tensor, token_ids
        return tf.nn.softmax(output_tensor), token_ids
