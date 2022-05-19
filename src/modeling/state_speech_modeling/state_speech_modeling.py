import sys

from src.configuration.deep_step import *
import tensorflow as tf
import datetime
from src.dataset_processing.training_data import *


def array_to_string(array):
    ss = ''
    for ele in array:
        ss += str(ele) + ','
    return ss


if __name__ == "__main__":
    # def remove_blanks(Z_i, Z_ni, A_i, A_ni, State_i, Theta_not_i):
    #     indexes = []
    #     for i in range(Z_ni.shape[0]):
    #         if Z_ni[i, 0, 1] == 102:
    #             indexes += [i]
    #     Z_i, Z_ni, A_i, A_ni, State_i, Theta_not_i = np.delete(Z_i, indexes, axis=0), np.delete(Z_ni, indexes, axis=0),\
    #                                                  np.delete(A_i, indexes, axis=0), np.delete(A_ni, indexes, axis=0), \
    #                                                  np.delete(State_i, indexes, axis=0), np.delete(Theta_not_i, indexes, axis=0)
    #     return Z_i, Z_ni, A_i, A_ni, State_i, Theta_not_i
    #
    # def get_messages():
    #     training_data.load_next_trn_file()
    #     if training_data.file_num == 2:
    #         print()
    #     Z_i, Z_ni, A_i, A_ni = training_data.get_messages_and_actions()
    #     State_i = training_data.get_state_i()
    #     Theta_not_i = training_data.get_theta_not_i()[:-1]
    #     Z_i, Z_ni, A_i, A_ni, State_i, Theta_not_i = remove_blanks(Z_i, Z_ni, A_i, A_ni, State_i, Theta_not_i)
    #     while Z_ni.shape[0] < batch_size + 1:
    #         training_data.load_next_trn_file()
    #         if training_data.file_num == 2:
    #             print()
    #         Z_it, Z_nit, A_it, A_nit = training_data.get_messages_and_actions()
    #         State_it = training_data.get_state_i()
    #         Theta_not_it = training_data.get_theta_not_i()[:-1]
    #         Z_i, Z_ni, A_i, A_ni = np.append(Z_i, Z_it, axis=0), np.append(Z_ni, Z_nit, axis=0), \
    #                                np.append(A_i, A_it, axis=0), np.append(A_ni, A_nit, axis=0)
    #         State_i, Theta_not_i = np.append(State_i, State_it, axis=0), np.append(Theta_not_i, Theta_not_it, axis=0)
    #         Z_i, Z_ni, A_i, A_ni, State_i, Theta_not_i = remove_blanks(Z_i, Z_ni, A_i, A_ni, State_i, Theta_not_i)
    #     return Z_i, Z_ni, A_i, A_ni, State_i, Theta_not_i

    pretrain_model.load_weights(PRETRAIN_TENSOR_MODEL)  # load from autoencoder
    # state_model.load_weights(STATE_TENSOR_MODEL)  # load from last checkpoint

    # EPOCHS = 100
    EPOCHS = 25
    # EXAMPLES_PER_EPOCH = 532
    EXAMPLES_PER_EPOCH = 2012 * 3
    EXAMPLES_PER_TEST = 25
    # EXAMPLES_PER_EPOCH = 2012
    # EXAMPLES_PER_TEST = 25

    training_data.load_next_batch()

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for j in range(EXAMPLES_PER_EPOCH):
            # tf.summary.trace_on(graph=True, profiler=True)

            training_data.load_next_batch(ssharp_data)
            # while training_data.tds.get_game() != 'chicken2':  # training only prisoners dilemma
            #     training_data.load_next_batch(ssharp_data)
            Z_i, Z_ni, compiled_state, Tar, state_org, theta_org = training_data.get_batch(ssharp_data)
            results = np.array(state_train_step2(compiled_state, Tar))

            # Tars, compiled_states = [], []
            # while len(Tars) < batch_size:
            #     training_data.load_next_batch(ssharp_data)
            #     while training_data.tds.get_game() != 'prisoners':
            #         training_data.load_next_batch(ssharp_data)
            #
            #     Z_i, Z_ni, compiled_state, Tar, state_org, theta_org = training_data.get_batch(ssharp_data)
            #
            #     for i in range(Tar.shape[0]):
            #         if training_data.contains_proposal(Tar[i]) or 6581 in Tar[i]:
            #             Tars += [Tar[i]]
            #             compiled_states += [compiled_state[i]]
            # Tar = np.array(Tars[:batch_size])
            # compiled_state = np.array(compiled_states[:batch_size])
            # results = np.array(state_train_step2(compiled_state, Tar))
            # for i in range(batch_size):
            #     print(training_data.convert_to_words(Tar[i], training_data.tds.get_game()))
            # Tar = ['lets play @ax', 'lets play @fair', '']
            # for i in range(batch_size):
            #     if i % 3 == 0:
            #         Tar[i] = np.append(training_data.convert_to_tokens('lets play @ax'), [0])
            #     elif i % 3 == 1:
            #         Tar[i] = np.append(training_data.convert_to_tokens('lets play @fair'), [0])
            #     elif i % 3 == 2:
            #         Tar[i] = np.append(training_data.convert_to_tokens('excellent'), [0])


            # if 'hicken' in training_data.tds.get_game():
            #
            #     game_output = open("ssm_outfile.csv", "w")
            #     out = training_data.tds.current_files[0] + '\n'
            #     for i in range(batch_size):
            #         out += "starting round," + str(i) + '\n'
            #         # generate speech from previous round data
            #         zi = training_data.convert_to_words(Z_i[i], training_data.tds.get_game())
            #         zni = training_data.convert_to_words(Z_ni[i], training_data.tds.get_game())
            #
            #         zi = re.sub('\[PAD\]', '', zi)
            #         zni = re.sub('\[PAD\]', '', zni)
            #         out += "S# Speech, " + zi + '\n'
            #         out += "Row Speech, " + zni + '\n'
            #         out += ',round,s0_1,s1_0,s1_1,s1_2,s1_3,s2_1,s3_1,s3_2,s3_3,s3_4,s3_5,s3_6,s4_1,s4_2,s4_3,s4_4,s4_5,s4_6,s4_7,s5_1,s5_2,s5_3,s5_4,s5_5,s5_6,s5_7,s5_8,s5_9,s5_10,s5_11,s5_12,s6_1,s6_2,s6_3,s6_4,s6_5,s6_6,s6_7,s6_8,s6_9,s6_10,s6_11,s6_12,s6_13,s7_1,s7_2,s7_3,s7_4,s7_5,s7_6,s7_7,s7_8,s7_9,s7_10,s7_11,s7_12,s8_1,s8_2,s8_3,s8_4,s8_5,s8_6,s8_7,s8_8,s8_9,s8_10,s8_11,s8_12,s8_13,s9_1,s9_2,s9_3,s9_4,s9_5,s9_6,s9_7,s9_8,s9_9,s9_10,s9_11,s9_12,s10_1,s10_2,s10_3,s10_4,s10_5,s10_6,s10_7,s10_8,s10_9,s10_10,s10_11,s10_12,s10_13\n'
            #         out += "S# State, " + array_to_string(state_org[i]) + '\n'
            #         out += "Row State, " + array_to_string(theta_org[i]) + '\n'
            #         out += "Compiled State, " + array_to_string(compiled_state[i]) + '\n\n'
            #
            #     game_output.write(out)
            #     game_output.close()
            #     sys.exit()

        for j in range(EXAMPLES_PER_TEST):
            training_data.load_next_batch('ssharp')
            Z_i, Z_ni, compiled_state, Tar, _, _ = training_data.get_batch('ssharp')
            results = np.array(state_test_step(compiled_state, Tar))
            if j == EXAMPLES_PER_TEST - 1:
                for i in range(1, batch_size):
                    # print('Original  : ', training_data.convert_to_words(Z_ni[i]))
                    print('Prediction:  [CLS]', training_data.convert_to_words(results[i], training_data.tds.get_game()), training_data.tds.get_game())
                    print('True      : ', training_data.convert_to_words(Tar[i, :-1], training_data.tds.get_game()), training_data.tds.get_game())
                    print()

        state_model.save_weights(STATE_TENSOR_MODEL)
        if epoch % 3 == 0:
            state_model.save_weights(STATE_TENSOR_MODEL + str(epoch))
        print(
            f'Epoch: {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            # f'Accuracy: {train_accuracy.result() * 100}, '
            # f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
            '\n\n\n'
        )
