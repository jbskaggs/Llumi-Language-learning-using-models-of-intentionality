import datetime
import time
from src.configuration.deep_step import *
from src.old.training_data_language_v3 import *
from src.dataset_processing.ssharp.gamesBf_v2 import *
import re

next_round = 0


def array_to_string(array):
    ss = ''
    for ele in array:
        ss += str(ele) + ','
    return ss


def set_up_theta(gameStr, me=0):
    bf = gamesBf(gameStr, me)
    mbrl = MBRL(gameStr, me)
    return bf, mbrl


def get_theta(z_i, z_ni, a_i, a_ni, bf, mbrl, game, logfile):
    listen = 1
    me = 0
    if round_num == 0:
        speech_state = bf.priorsOnly
    else:
        z_ni = [findProposal(z_ni)]
        z_i = [findProposal(z_i)]
        observations_zMe = bf.findOtherMessages(z_ni)
        observations_zYou = bf.findOtherMessages(z_i)

        BelNextState = bf.calculateBelNextState(observations_zMe[0], listen)
        # writer.writerow([str(i + 1)] + BelNextState)  # State 1

        BelbarCurrentState = bf.calculateBelbarCurrentState(observations_zMe[0], observations_zYou[0], me, listen)
        # writer.writerow([str(i)] + BelbarCurrentState)  # State 2

        BelCurrentState = bf.calculateBelCurrentState(a_ni)
        # writer.writerow([str(i)] + BelCurrentState)  # State 3

        if me == 0:
            mbrl.update(int(a_ni),
                        int(a_i))  # Update the MBRL expert's counts, Q-values, and current state
        else:
            mbrl.update(int(a_i),
                        int(a_ni))  # Update the MBRL expert's counts, Q-values, and current state
        bf.P_A_S = mbrl.updatePAS(bf.P_A_S)  # update bf.P_A_S
        bf.P_S_A_A_S = mbrl.updateSAS(bf.P_S_A_A_Scopy)  # update bf.P_S_A_A_S

        speech_state = bf.calculateBelbarNextState(a_ni, a_i, me)
        # writer.writerow([str(i + 1)] + BelbarNextState)  # State 4

    speech_state = np.append(np.zeros(1)+round_num, np.array([speech_state], dtype=np.float))
    priors = speech_state.reshape(1, speech_state.shape[0])
    game_output = open(logfile, "a")
    out = "Row State," + array_to_string(priors[0]) + '\n'
    game_output.write(out)
    game_output.close()
    return training_data.tds.probability_profile_to_reformed_state_representation(priors, game)


def get_state(zeros, state, game, round, logfile):
    speech_state = np.append(np.zeros(1)+round_num, np.array(training_data.tds.get_reformed_state(game).loc[[state]]))
    game_output = open(logfile, "a")
    out = "S# State, " + array_to_string(state) + '\n'
    out += "S# State, " + array_to_string(speech_state) + '\n'
    game_output.write(out)
    game_output.close()
    return speech_state.reshape(1, speech_state.shape[0])


def generate_speech(z_i, z_ni, a_i, a_ni, s_i, t_ni, proposals_i, proposals_not_i, game='prisoners', agent=1, logfile="output.csv"):
    Z_i = training_data.tds.format_text([z_i]).reshape((1, word_buffer_len))
    Z_ni = training_data.tds.format_text([z_ni]).reshape((1, word_buffer_len))
    A_i = np.array([a_i])
    A_ni = np.array([a_ni])
    A = np.concatenate((A_i.reshape((A_i.shape[0]), 1), A_ni.reshape((A_ni.shape[0]), 1)), axis=1)
    State_i = s_i
    Theta_not_i = t_ni
    proposals_i = np.array([proposals_i])
    proposals_not_i = np.array([proposals_not_i])
    compiled_state = np.concatenate([A, State_i, Theta_not_i, proposals_i, proposals_not_i], axis=1)
    # compiled_state = np.concatenate([A, State_i, Theta_not_i], axis=1)  # I removed proposals because they don't seem to do anything

    game_output = open(logfile, "a")
    out = "Compiled State," + array_to_string(compiled_state[0]) + '\n'
    game_output.write(out)
    game_output.close()

    target = np.zeros_like(Z_ni)
    target[0, 0] = 101
    # if agent == 1: used in first user study test
    #     results = live_test_step_random_greedy_thresholding(Z_i, Z_ni, np.array(compiled_state, dtype=float), target, blanks_discount=.8, threshold=.002, logfile=logfile)
    # else:
    #     results = live_test_step_random_greedy_top_of_batch(Z_i, Z_ni, np.array(compiled_state, dtype=float), target, blanks_discount=.141, logfile=logfile)
    if agent == 1:
        results = live_test_step_random_greedy_thresholding(Z_i, Z_ni, np.array(compiled_state, dtype=float), target, game, blanks_discount=.6, threshold=.000005, logfile=logfile)
    else:
        results = live_test_step_random_greedy_thresholding(Z_i, Z_ni, np.array(compiled_state, dtype=float), target, game, blanks_discount=.6, threshold=.001, logfile=logfile)
        # results = live_test_step_random_greedy_top_of_batch(Z_i, Z_ni, np.array(compiled_state, dtype=float), target, blanks_discount=.1, logfile=logfile)

    results = training_data.convert_to_words(results, game)
    results = training_data.remove_padding(results)
    return results


def convert_state2standard_state(state, game):
    if game == 'prisoners':
        if 's5' in state:
            state = re.sub('s5', 's7', state)
        elif 's6' in state:
            state = re.sub('s6', 's8', state)
        elif 's7' in state:
            state = re.sub('s7', 's5', state)
        elif 's8' in state:
            state = re.sub('s8', 's6', state)
    elif game == 'chicken2':
        if 's3' in state:
            state = re.sub('s3', 's7', state)
        elif 's4' in state:
            state = re.sub('s4', 's8', state)
        elif 's5' in state:
            state = re.sub('s5', 's3', state)
        elif 's6' in state:
            state = re.sub('s6', 's4', state)
        elif 's7' in state:
            state = re.sub('s7', 's5', state)
        elif 's8' in state:
            state = re.sub('s8', 's6', state)
        elif 's9' in state:
            state = re.sub('s9', 's11', state)
        elif 's10' in state:
            state = re.sub('s10', 's12', state)
        elif 's11' in state:
            state = re.sub('s11', 's9', state)
        elif 's12' in state:
            state = re.sub('s12', 's10', state)
    elif game == 'endless':
        pass
    elif game == 'chickenalt':
        pass
    return state


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    game = sys.argv[3]
    agent = sys.argv[4]
    player = "player" + output_file.split('/')[-1][-12:-11]
    the_date_time = str(datetime.datetime.now())
    logfile = "../../../out/out_" + game + "_" + player + "_JBS" + agent + "_" + the_date_time + ".csv"

    print("Starting speech agent JBS" + agent + "...")
    print("reading from: " + input_file)
    print("writing to: " + output_file)
    print("Playing: " + game)
    print()

    # set up tensorflow
    state_model.load_weights(STATE_TENSOR_MODEL)
    # generate_speech(' ', 'Hello World', 1, 1, np.zeros(162), np.zeros(162))
    # generate_speech(' ', 'Hello World', 1, 1, np.zeros(162), np.zeros(162))

    # set up theta calculations
    bf, mbrl = set_up_theta(game)

    while True:
        # os.system("scp jonathanskaggs@"+macbook_ip+":/Users/jonathanskaggs/IdeaProjects/RepeatedPlaySpeechActInterface/src/theJBS/stateInfo.txt stateInfo.txt")
        f = open(input_file, "r")
        lines = f.readlines()
        f.close()

        if int((lines[0].split(':')[1][:-1])) > 999:
            sys.exit()

        if len(lines) > 3:
            round_num, actions, speech, state = lines[:4]
            round_num = int((round_num.split(':')[1][:-1]))
            speech_zni, speech_zi = speech[:-1].split(':')[1:]
            action_ani, action_ai = actions[:-1].split(':')[1:]
            action_ai = int(action_ai)
            action_ani = int(action_ani)
            state = convert_state2standard_state(state.split(':')[-1][:-1], game)

            if next_round == round_num:
                game_output = open(logfile, "a")
                out = "starting round," + str(next_round) + "\n"
                # generate speech from previous round data
                out += "S# Speech, " + speech_zi + '\n'
                out += "Row Speech, " + speech_zni + '\n'
                game_output.write(out)
                game_output.close()
                theta = get_theta(speech_zi, speech_zni, action_ai, action_ani, bf, mbrl, game, logfile)
                state = get_state(np.zeros_like(theta, dtype=float), state, game, round_num, logfile)
                proposal_i, proposal_not_i = training_data.get_proposals_from_text([speech_zi, speech_zni], game)
                new_speech = generate_speech(speech_zi, speech_zni, action_ai, action_ani, state, theta, proposal_i, proposal_not_i, game, int(agent), logfile=logfile)

                # prepare generated speech to be read by file
                g = open(output_file, "w")
                ss = "round: " + str(round_num) + '\n'
                ss += "speech: " + new_speech + '\n'

                print(ss + "\n")

                # write our generated speech to a file so spp can read it
                g.write(ss)
                g.close()
                # os.system("scp speech.txt jonathanskaggs@"+macbook_ip+":/Users/jonathanskaggs/IdeaProjects/RepeatedPlaySpeechActInterface/src/theJBS/speech.txt")

                # Update the round we are now going to look for next time we open the file
                next_round += 1
            else:
                time.sleep(1)
        else:
            time.sleep(1)
