import numpy as np

from src.configuration.configuration import *
import tensorflow as tf
import src.model.model as tensor_model

# Create an instance of the model
pretrain_model = tensor_model.autoencoder(batch_size, vocab_size, embedding_size, word_buffer_len, num_layers, num_heads)
# midtrain_model = tensor_model.mid_model(pretrain_model, batch_size, word_buffer_len)
if not autoencoder_bool:
    conversation_model = tensor_model.skaggs_model(pretrain_model, batch_size, word_buffer_len)
    state_model = tensor_model.state_speech(pretrain_model, batch_size, word_buffer_len)
autoencoder_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
state_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
conversation_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam(learning_rate)

# Define our metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


###### Autoencoder ######

@tf.function
def autoencoder_train_step(inputs, outputs, mask):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, token_ids, vocab_loss = pretrain_model(inputs, outputs[:, :-1], mask, training=True)
        loss = autoencoder_loss_object(outputs[:, 1:], predictions)
        loss += 39 * autoencoder_loss_object(tf.cast(outputs[:, 1:], tf.float32) * tf.cast(mask, tf.float32),
                                             tf.cast(predictions, tf.float32) *
                                             tf.cast(tf.reshape(mask, [batch_size, word_buffer_len, 1]),
                                                     tf.float32))
        loss += vocab_loss

    gradients = tape.gradient(loss, pretrain_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, pretrain_model.trainable_variables))

    train_loss(loss)
    train_accuracy(outputs[:, 1:], predictions)
    return token_ids, vocab_loss


# @tf.function
# def autoencoder_test_step(inputs):
#     # training=False is only needed if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     predictions, token_ids = pretrain_model(inputs, training=False)
#     loss = autoencoder_loss_object(inputs, predictions)
#
#     test_loss(loss)
#     test_accuracy(inputs, predictions)


###### State Model ######

@tf.function
def state_train_step(state, target):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        error = state_model((state, target[:, 1:]), training=True)
        # loss = loss_object(target[:, 1:], predictions)

    gradients = tape.gradient(error, state_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, state_model.trainable_variables))

    train_loss(error)
    # train_accuracy(target[:, 1:], predictions)
    return error


@tf.function
def state_train_step2(state, target):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, token_ids = state_model((state, target[:, :-1]), training=True)
        loss = state_loss_object(target[:, 1:], predictions)

    gradients = tape.gradient(loss, state_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, state_model.trainable_variables))

    train_loss(loss)
    # train_accuracy(target[:, 1:], predictions)
    return loss


@tf.function
def state_test_step(state, target):
    predictions, token_ids = state_model((state, target[:, :-1]), training=True)

    # train_loss(error)
    test_accuracy(target[:, 1:], predictions)
    return token_ids


###### Conversational Model ######

@tf.function
def train_step(z_i, z_ni, state=None, target=None):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions, token_ids = conversation_model((z_i, z_ni, state, target[:, :-1]), training=True)
        loss = conversation_loss_object(target[:, 1:], predictions)

    gradients = tape.gradient(loss, conversation_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, conversation_model.trainable_variables))

    train_loss(loss)
    train_accuracy(target[:, 1:], predictions)
    return token_ids


@tf.function
def test_step(z_i, z_ni, state=None, target=None):
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions, token_ids = conversation_model((z_i, z_ni, state, target[:, :-1]), training=True)
    loss = conversation_loss_object(target[:, 1:], predictions)

    test_loss(loss)
    test_accuracy(target[:, 1:], predictions)
    return token_ids


###### Live Model ######

def live_test_step(z_i, z_ni, state, target):
    boolean = tf.linalg.band_part(tf.ones((word_buffer_len, word_buffer_len)), -1, 0)
    state = np.tile(state, (batch_size, 1))
    target = np.tile(target, (batch_size, 1))
    for i in range(int(word_buffer_len)):
        predictions, token_ids = state_model((state, target), training=False)
        target = tf.concat((tf.reshape([101.0] * batch_size, (batch_size, 1)),
                            tf.reshape(tf.cast(token_ids, tf.float32) * boolean[i], (batch_size, word_buffer_len))[:,
                            :-1]), axis=1)
    return target[0]


NUM_CHOICES = 2


def live_test_step_random_greedy(z_i, z_ni, state, target):
    final_choice = 0
    state = np.tile(state, (batch_size, 1))
    target = np.tile(target, (batch_size, 1))
    for i in range(int(word_buffer_len) - 1):
        predictions, token_ids = state_model((state, target), training=False)
        final_predictions = np.array(predictions)
        final_predictions[:, i, final_choice] = 0
        ind = np.argpartition(final_predictions[0, i], -NUM_CHOICES)[
              -NUM_CHOICES:]  # get the top results from each prediction
        score = final_predictions[0, i][ind]
        if i == 0:
            for j in range(len(ind)):
                if ind[j] == 102:
                    score[j] /= 5

        final_choice = choose_probabilistically(ind, score)
        if final_choice == 5138 and np.sum(state[0, -4:]) == 0:  # 5318 is the word accept
            target = np.zeros_like(target)
            break
        if final_choice == 0:  # if it is [PAD] then it should be done
            break
        if final_choice == 102:  # if it is [EOS] then it should be done
            target[:, i + 1] = final_choice
            break
        target[:, i + 1] = final_choice
    return target[0]


def live_test_step_random_greedy_40_peaks(z_i, z_ni, state, target):
    final_choice = 0
    state = np.tile(state, (batch_size, 1))
    target = np.tile(target, (batch_size, 1))
    for i in range(int(word_buffer_len) - 1):
        predictions, token_ids = state_model((state, target), training=False)
        final_predictions = np.array(predictions)
        final_predictions[:, i, final_choice] = 0  # can't be the same word as the previous word
        if i == 0:
            NUM_CHOICES = batch_size
            ind = np.argpartition(final_predictions[0, i], -NUM_CHOICES)[
                  -NUM_CHOICES:]  # get the top results from each prediction
            final_scores = final_predictions[0, i][ind]
        else:
            NUM_CHOICES = 1
            ind = np.argpartition(final_predictions[:, i], -NUM_CHOICES, axis=1)[:,
                  -NUM_CHOICES:]  # get the top results from each prediction
            for j in range(len(final_scores)):
                if target[j, i] == 102 or target[j, i] == 0:
                    ind[j] = 0
                else:
                    final_scores[j] *= final_predictions[j,  i, ind[j]]

        # if final_choice == 102:  # if it is [EOS] then it should be done
        #     target[:, i + 1] = final_choice
        #     break
        target[:, i + 1] = ind.T

    score_args = np.flip(np.argsort(final_scores))

    for i in range(batch_size):
        print('Option ' + str(i) + ' : Score ', f'{final_scores[score_args[i]]*100:.3f}' + ' : ', training_data.remove_padding(training_data.convert_to_words(target[score_args[i]], training_data.tds.get_game())))
    return target[score_args[0]]


def live_test_step_random_greedy_40_samples(z_i, z_ni, state, target, blanks_discount=1.0):
    final_choice, final_scores = 0, np.ones(batch_size)
    NUM_CHOICES = 25
    state = np.tile(state, (batch_size, 1))
    target = np.tile(target, (batch_size, 1))
    for i in range(int(word_buffer_len) - 1):
        predictions, token_ids = state_model((state, target), training=False)
        final_predictions = np.array(predictions)
        final_predictions[:, i, final_choice] = 0  # can't be the same word as the previous word
        ind = np.argpartition(final_predictions[:, i], -NUM_CHOICES, axis=1)[:,
              -NUM_CHOICES:]  # get the top results from each prediction
        inds = np.zeros(batch_size)
        for j in range(len(final_scores)):
            scores = final_predictions[j, i, ind[j]]
            if i == 0:  # discount not saying anything
                for k in range(NUM_CHOICES):
                    if ind[j, k] == 102:
                        scores[k] *= blanks_discount
            inds[j] = choose_probabilistically(ind[j], scores)
            if target[j, i] == 102 or target[j, i] == 0:
                inds[j] = 0
            else:
                final_scores[j] *= final_predictions[j,  i, int(inds[j] + .5)]


        # if final_choice == 102:  # if it is [EOS] then it should be done
        #     target[:, i + 1] = final_choice
        #     break
        target[:, i + 1] = inds.T

    return target, final_scores


def live_test_step_random_greedy_120_samples(z_i, z_ni, state, target_in, logfile):
    target, final_scores = np.array([[]]), np.array([[]])
    for j in range(3):
        t, f = live_test_step_random_greedy_40_samples(z_i, z_ni, state, target_in)
        if j == 0:
            target = t
            final_scores = f
        else:
            target = np.append(target, t, axis=0)
            final_scores = np.append(final_scores, f, axis=0)

        # target = np.append(target, t, axis=1)
        # final_scores = np.append(final_scores, f, axis=0)
    score_args = np.flip(np.argsort(final_scores))

    file = open(logfile, 'a')
    for i in range(batch_size*3):
        ss = 'Option ' + str(i) + ' : Score ' + f'{final_scores[score_args[i]]*100:.3f}' + ' : ' + training_data.remove_padding(training_data.convert_to_words(target[score_args[i]], training_data.tds.get_game()))
        print(ss)
        file.write(ss + '\n')
    file.write('\n')
    file.close()

    return target[score_args[0]]


def live_test_step_random_greedy_thresholding(z_i, z_ni, state, target_in, game, blanks_discount=.5, threshold=.002, logfile="outfile.csv"):
    target, final_scores = live_test_step_random_greedy_40_samples(z_i, z_ni, state, target_in, blanks_discount)

    score_args = np.arange(batch_size)

    file = open(logfile, 'a')
    for i in range(batch_size):
        ss = 'Option ' + str(
            i) + ' : Score ' + f'{final_scores[score_args[i]] * 100:.3f}' + ' : ' + training_data.remove_padding(
            training_data.convert_to_words(target[score_args[i]], game))
        print(ss)
        file.write(ss + '\n')

    for i in range(10):
        # ind = np.random.randint(batch_size-1)
        print('ind: ' + str(i))
        file.write('ind: ' + str(i) + '\n')
        if int(np.sum(state[0, -4:])) == 0 and 5138 in target[i]:  # can't accept when there is nothing to accept
            pass
        elif final_scores[i] > threshold:
            return target[i]
    file.write('\n')
    file.close()

    score_args = np.flip(np.argsort(final_scores))

    return target[score_args[0]]


def live_test_step_random_greedy_top_of_batch(z_i, z_ni, state, target_in, blanks_discount=.5, logfile="outfile.csv"):
    target, final_scores = live_test_step_random_greedy_40_samples(z_i, z_ni, state, target_in, blanks_discount)

    score_args = np.arange(batch_size)

    file = open(logfile, 'a')
    for i in range(batch_size):
        if target[i, 1] == 102:
            final_scores[i] = final_scores[i] * blanks_discount

        ss = 'Option ' + str(
            i) + ' : Score ' + f'{final_scores[score_args[i]] * 100:.3f}' + ' : ' + training_data.remove_padding(
            training_data.convert_to_words(target[score_args[i]], training_data.tds.get_game()))
        print(ss)
        file.write(ss + '\n')

    file.write('\n')
    file.close()

    score_args = np.flip(np.argsort(final_scores))

    return target[score_args[0]]


def choose_probabilistically(choices, scores):
    ind = np.random.choice(np.arange(len(choices)), p=scores/np.sum(scores))
    return choices[ind]


def live_test_step_beam_search(z_i, z_ni, state, target, using_probability=True):
    boolean = tf.linalg.band_part(tf.ones((word_buffer_len, word_buffer_len)), -1, 0)
    z_i, z_ni = np.tile(z_i, (batch_size, 1)), np.tile(z_ni, (batch_size, 1))
    state = np.tile(state, (batch_size, 1))
    target = np.tile(target, (batch_size, 1))
    scores = np.ones(batch_size)
    for i in range(int(word_buffer_len) - 1):
        predictions, token_ids = state_model((state, target), training=False)
        if i == 0:
            ind = np.argpartition(predictions[0, i], -batch_size)[-batch_size:]  # get the top results from each prediction
            score = np.array(predictions[0, i])[ind]
            target[:, i + 1] = ind
            scores = scores * score
        else:
            new_targets, new_scores = [], []
            for j, pred in enumerate(predictions):
                tmp_target = np.tile(target[j], (3, 1))
                ind = np.argpartition(pred[i], -3)[-3:]  # get the top results from each prediction
                score = np.array([pred[i, ind[0]], pred[i, ind[1]], pred[i, ind[2]]])
                tmp_target[:, i + 1] = ind
                tmp_score = np.tile(scores[j], [3]) * score
                new_targets += [tmp_target]
                new_scores += [tmp_score]
            new_targets = np.array(new_targets)
            new_targets = np.reshape(new_targets, [new_targets.shape[0] * new_targets.shape[1], new_targets.shape[2]])
            new_scores = np.array(new_scores)
            new_scores = np.reshape(new_scores, [new_scores.shape[0] * new_scores.shape[1]])
            idx = np.argpartition(new_scores, -batch_size)[-batch_size:]
            target = new_targets[idx]
            scores = new_scores[idx]
    if using_probability:
        return choose_probabilistically(target, scores)
    return target[np.argmax(scores)]
