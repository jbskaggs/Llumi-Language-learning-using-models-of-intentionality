import random

import numpy as np
import os
import pandas as pd
import re
import math
import csv
from transformers import BertTokenizer

DATASET_PATH = "/home/jskaggs93/Datasets/"


class TrainingData:
    def __init__(self, word_buffer_len, tokenizer):
        self.tokenizer = tokenizer

        # preprocessing variables
        self.logfile = 'SSharpDataset/Logs/speech_logs/Logs/Logs/'
        self.logfile48 = 'SSharpDataset/Logs/speech_logs/Logs_48subjects/Logs/'
        self.logfileTR = 'SSharpDataset/Logs/speech_logs/org_experiment/TennomResultsTalk/'
        self.composite = 'SSharpDataset/Logs/speech_logs/LogsComposite_Logs/'
        self.composite2 = 'SSharpDataset/Logs/speech_logs/Composite_Logs_4/'
        self.speechStateConversation = DATASET_PATH + 'SSharpDataset/SpeechStateConversion/'

        # global variables
        self.word_buffer_len = word_buffer_len
        self.possible_chats_expanded = np.array(pd.read_csv(DATASET_PATH + 'SSharpDataset/Logs/personalities_speech_acts/possible_chats_expanded.csv', header=None))
        for i in range(self.possible_chats_expanded.shape[0]):
            self.possible_chats_expanded[i] = ['None' if x is np.nan else x for x in self.possible_chats_expanded[i]]

        # language modeling variables
        self.LOGS = DATASET_PATH + 'SSharpDataset/Logs/speech_logs/Composite_Logs_4/'
        self.PROB_PROFILE = DATASET_PATH + 'SSharpDataset/ProbabilityProfiles/'
        self.VITERBI_LOG = DATASET_PATH + 'SSharpDataset/ViterbiLogs2/'
        self.TRAINING_FILES = self.__get_training_files()
        self.current_files = ['', '', '']  # This should be a tuple of three.  First is the log of actions,
        # Second is the log of the state of player i, and lastly is the file that describes the
        # probability distribution over the state of player not i
        self.file_num = -1
        self.load_next_trn_file()  # Sets the self.current_file to a random file in the training set
        self.load_speech_state_sets()

    def format_text(self, texts):
        trn_datas = []
        for text in texts:
            trn_data = np.zeros((1, self.word_buffer_len))
            if text == "No Messages Sent" or pd.isna(text):
                text = ''
            tokens = self.tokenizer.encode(text)
            for j, token in enumerate(tokens[:self.word_buffer_len]):
                trn_data[0, j] = token
            trn_datas += [trn_data]
        return np.array(trn_datas)

    def load_speech_state_sets(self):
        self.prisoners_reformed_state = pd.read_csv(self.speechStateConversation + 'prisoners_speech_state_conversion.csv', index_col=0)
        self.chicken_reformed_state = pd.read_csv(self.speechStateConversation + 'chicken_speech_state_conversion.csv', index_col=0)
        self.blocks_reformed_state = pd.read_csv(self.speechStateConversation + 'blocks_speech_state_conversion.csv', index_col=0)
        self.endless_reformed_state = pd.read_csv(self.speechStateConversation + 'endless_speech_state_conversion.csv', index_col=0)
        self.chickenalt_reformed_state = pd.read_csv(self.speechStateConversation + 'chickenalt_speech_state_conversion.csv', index_col=0)

    def get_reformed_state(self, game=None):
        if game is None:
            game = self.get_game().lower()
        if game == 'blocks2':
            return self.blocks_reformed_state
        elif game == 'chicken2':
            return self.chicken_reformed_state
        elif game == 'prisoners':
            return self.prisoners_reformed_state
        elif game == 'chickenalt':
            return self.chickenalt_reformed_state
        elif game == 'endless':
            return self.endless_reformed_state

    @staticmethod
    def get_state_transitions(game):
        if game.lower() == 'blocks2':
            state_transitions = ['s0_1', 's1_0', 's1_1', 's1_3', 's1_4', 's1_2', 's1_5', 's1_6', 's1_7', 's1_8', 's2_1',
                                 's5_1', 's5_2', 's5_3', 's5_4', 's5_5', 's5_6', 's5_7', 's5_8', 's5_9', 's5_10', 's5_11', 's5_12',
                                 's6_1', 's6_2', 's6_3', 's6_4', 's6_5', 's6_6', 's6_7', 's6_8', 's6_9', 's6_10', 's6_11', 's6_12', 's6_13',
                                 's7_1', 's7_1', 's7_2', 's7_2', 's7_3', 's7_3', 's7_4', 's7_4', 's7_5', 's7_5', 's7_6', 's7_6',
                                 's8_1', 's8_1', 's8_2', 's8_2', 's8_3', 's8_3', 's8_4', 's8_4', 's8_5', 's8_5', 's8_6', 's8_6', 's8_7',
                                 's3_1', 's3_1', 's3_2', 's3_2', 's3_3', 's3_3', 's3_4', 's3_4', 's3_5', 's3_5', 's3_6', 's3_6',
                                 's4_1', 's4_1', 's4_2', 's4_2', 's4_3', 's4_3', 's4_4', 's4_4', 's4_5', 's4_5', 's4_6', 's4_6', 's4_7',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000']
        elif game.lower() == 'chicken2':
            state_transitions = ['s0_1', 's1_0', 's1_1', 's1_2', 's1_3', '0000', '0000', '0000', '0000', '0000', 's2_1',
                                 's3_1', 's3_1', 's3_2', 's3_2', 's3_3', 's3_3', 's3_4', 's3_4', 's3_5', 's3_5', 's3_6', 's3_6',
                                 's4_1', 's4_1', 's4_2', 's4_2', 's4_3', 's4_3', 's4_4', 's4_4', 's4_5', 's4_5', 's4_6', 's4_6', 's4_7',
                                 's5_1', 's5_1', 's5_2', 's5_2', 's5_3', 's5_3', 's5_4', 's5_4', 's5_5', 's5_5', 's5_6', 's5_6',
                                 's6_1', 's6_1', 's6_2', 's6_2', 's6_3', 's6_3', 's6_4', 's6_4', 's6_5', 's6_5', 's6_6', 's6_6', 's6_7',
                                 's7_1', 's7_1', 's7_2', 's7_2', 's7_3', 's7_3', 's7_4', 's7_4', 's7_5', 's7_5', 's7_6', 's7_6',
                                 's8_1', 's8_1', 's8_2', 's8_2', 's8_3', 's8_3', 's8_4', 's8_4', 's8_5', 's8_5', 's8_6', 's8_6', 's8_7',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 's9_1', 's9_2', 's9_3', 's9_4', 's9_5', 's9_6', 's9_7', 's9_8', 's9_9', 's9_10', 's9_11', 's9_12',
                                 's10_1', 's10_2', 's10_3', 's10_4', 's10_5', 's10_6', 's10_7', 's10_8', 's10_9', 's10_10', 's10_11', 's10_12', 's10_13',
                                 's11_1', 's11_2', 's11_3', 's11_4', 's11_5', 's11_6', 's11_7', 's11_8', 's11_9', 's11_10', 's11_11', 's11_12',
                                 's12_1', 's12_2', 's12_3', 's12_4', 's12_5', 's12_6', 's12_7', 's12_8', 's12_9', 's12_10', 's12_11', 's12_12', 's12_13']
        elif game.lower() == 'prisoners':
            state_transitions = ['s0_1', 's1_0', 's1_1', 's1_2', 's1_3', '0000', '0000', '0000', '0000', '0000', 's2_1',
                                 's3_1', 's3_1', 's3_2', 's3_2', 's3_3', 's3_3', 's3_4', 's3_4', 's3_5', 's3_5', 's3_6', 's3_6',
                                 's4_1', 's4_1', 's4_2', 's4_2', 's4_3', 's4_3', 's4_4', 's4_4', 's4_5', 's4_5', 's4_6', 's4_6', 's4_7',
                                 's9_1', 's9_2', 's9_3', 's9_4', 's9_5', 's9_6', 's9_7', 's9_8', 's9_9', 's9_10', 's9_11', 's9_12',
                                 's10_1', 's10_2', 's10_3', 's10_4', 's10_5', 's10_6', 's10_7', 's10_8', 's10_9', 's10_10', 's10_11', 's10_12', 's10_13',
                                 's7_1', 's7_2', 's7_3', 's7_4', 's7_5', 's7_6', 's7_7', 's7_8', 's7_9', 's7_10', 's7_11', 's7_12',
                                 's8_1', 's8_2', 's8_3', 's8_4', 's8_5', 's8_6', 's8_7', 's8_8', 's8_9', 's8_10', 's8_11', 's8_12', 's8_13',
                                 's5_1', 's5_2', 's5_3', 's5_4', 's5_5', 's5_6', 's5_7', 's5_8', 's5_9', 's5_10', 's5_11', 's5_12',
                                 's6_1', 's6_2', 's6_3', 's6_4', 's6_5', 's6_6', 's6_7', 's6_8', 's6_9', 's6_10', 's6_11', 's6_12', 's6_13',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000',
                                 '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000', '0000']
        else:
            raise Exception()
        return state_transitions

    def __probability_profile_to_state_representation(self, table):
        state_transitions = self.get_state_transitions(self.get_game())

        num_state_transitions, num_states = np.array(table).shape
        new_table = [(np.arange(num_state_transitions)).astype('int').astype('float')]
        for s in state_transitions:
            if s == '0000':
                new_table += [np.zeros(num_state_transitions)]
            else:
                new_table += [np.array(table[s])]
        new_table = np.array(new_table).transpose()

        return new_table

    def probability_profile_to_reformed_state_representation(self, table, game=None):
        state_transitions = self.get_reformed_state(game)
        zeros = np.zeros((1, state_transitions.shape[1]))
        state_transitions = np.append(zeros, np.array(state_transitions), axis=0)
        zeros = np.zeros((state_transitions.shape[0], 1))
        state_transitions = np.append(zeros, np.array(state_transitions), axis=1)
        state_transitions[0, 0] = 1
        new_table = np.matmul(table, state_transitions)
        return np.array(new_table)

    def pp2state(self, table, game):
        table = pd.DataFrame(np.array(table), columns=self.get_original_states(game))
        state_transitions = self.get_state_transitions(game)

        num_state_transitions, num_states = np.array(table).shape
        new_table = [(np.arange(num_state_transitions)).astype('int').astype('float')]
        for s in state_transitions:
            if s == '0000':
                new_table += [np.zeros(num_state_transitions)]
            else:
                new_table += [np.array(table[s])]
        new_table = np.array(new_table).transpose()

        return new_table

    @staticmethod
    def get_original_states(game):
        if game.lower() == 'blocks2':
            state_transitions = ['s0_1', 's1_0', 's1_1', 's1_2', 's1_3', 's1_4', 's1_5', 's1_6', 's1_7', 's1_8', 's2_1',
                                 's3_1', 's3_2', 's3_3', 's3_4', 's3_5', 's3_6', 's4_1', 's4_2', 's4_3', 's4_4', 's4_5', 's4_6', 's4_7',
                                 's5_1', 's5_2', 's5_3', 's5_4', 's5_5', 's5_6', 's5_7', 's5_8', 's5_9', 's5_10', 's5_11', 's5_12',
                                 's6_1', 's6_2', 's6_3', 's6_4', 's6_5', 's6_6', 's6_7', 's6_8', 's6_9', 's6_10', 's6_11', 's6_12', 's6_13',
                                 's7_1', 's7_2', 's7_3', 's7_4', 's7_5', 's7_6', 's8_1', 's8_2', 's8_3', 's8_4', 's8_5', 's8_6', 's8_7']
        elif game.lower() == 'chicken2':
            state_transitions = ['s0_1', 's1_0', 's1_1', 's1_2', 's1_3', 's2_1', 's3_1', 's3_2', 's3_3', 's3_4', 's3_5', 's3_6',
                                 's4_1', 's4_2', 's4_3', 's4_4', 's4_5', 's4_6', 's4_7', 's5_1', 's5_2', 's5_3', 's5_4', 's5_5', 's5_6',
                                 's6_1', 's6_2', 's6_3', 's6_4', 's6_5', 's6_6', 's6_7', 's7_1', 's7_2', 's7_3', 's7_4', 's7_5', 's7_6',
                                 's8_1', 's8_2', 's8_3', 's8_4', 's8_5', 's8_6', 's8_7',
                                 's9_1', 's9_2', 's9_3', 's9_4', 's9_5', 's9_6', 's9_7', 's9_8', 's9_9', 's9_10', 's9_11', 's9_12',
                                 's10_1', 's10_2', 's10_3', 's10_4', 's10_5', 's10_6', 's10_7', 's10_8', 's10_9', 's10_10', 's10_11', 's10_12', 's10_13',
                                 's11_1', 's11_2', 's11_3', 's11_4', 's11_5', 's11_6', 's11_7', 's11_8', 's11_9', 's11_10', 's11_11', 's11_12',
                                 's12_1', 's12_2', 's12_3', 's12_4', 's12_5', 's12_6', 's12_7', 's12_8', 's12_9', 's12_10', 's12_11', 's12_12', 's12_13']
        elif game.lower() == 'prisoners':
            state_transitions = ['s0_1', 's1_0', 's1_1', 's1_2', 's1_3', 's2_1', 's3_1', 's3_2', 's3_3', 's3_4', 's3_5', 's3_6',
                                 's4_1', 's4_2', 's4_3', 's4_4', 's4_5', 's4_6', 's4_7',
                                 's5_1', 's5_2', 's5_3', 's5_4', 's5_5', 's5_6', 's5_7', 's5_8', 's5_9', 's5_10', 's5_11', 's5_12',
                                 's6_1', 's6_2', 's6_3', 's6_4', 's6_5', 's6_6', 's6_7', 's6_8', 's6_9', 's6_10', 's6_11', 's6_12', 's6_13',
                                 's7_1', 's7_2', 's7_3', 's7_4', 's7_5', 's7_6', 's7_7', 's7_8', 's7_9', 's7_10', 's7_11', 's7_12',
                                 's8_1', 's8_2', 's8_3', 's8_4', 's8_5', 's8_6', 's8_7', 's8_8', 's8_9', 's8_10', 's8_11', 's8_12', 's8_13',
                                 's9_1', 's9_2', 's9_3', 's9_4', 's9_5', 's9_6', 's9_7', 's9_8', 's9_9', 's9_10', 's9_11', 's9_12',
                                 's10_1', 's10_2', 's10_3', 's10_4', 's10_5', 's10_6', 's10_7', 's10_8', 's10_9', 's10_10', 's10_11', 's10_12', 's10_13']
        else:
            raise Exception()
        return state_transitions

    def get_game(self):
        return self.current_files[2].split('_')[2].lower()

    def get_player(self):
        return self.current_files[2].split('_')[3]

    def get_partner(self):
        return self.current_files[2].split('_')[4]

    def get_me(self):
        return self.current_files[2].split('_')[5][0]

    def __get_training_files(self):
        trn_files = []
        for me in [0, 1]:
            for dir in os.listdir(self.LOGS):
                id, game, player0, player1 = dir.split('_')
                player1 = player1[:-4]
                activity_log = self.LOGS + id + '_' + game + '_' + player0 + '_' + player1 + '.csv'
                probability_profile = self.PROB_PROFILE + 'probs_' + id + '_' + game + '_' + player0 + '_' + player1 + '_' + str(1 - me) + '.csv'
                state_of_player_i_log = self.PROB_PROFILE + 'probs_' + id + '_' + game + '_' + player0 + '_' + player1 + '_' + str(me) + '.csv'
                # state_of_player_i_log = self.VITERBI_LOG + id + '_' + game + '_' + player0 + '_' + player1 + '_' + str(me) + '.csv'
                trn_files += [[activity_log, state_of_player_i_log, probability_profile]]
        return np.array(trn_files)

    def load_next_trn_file(self):
        # rand_num = round(np.random.random() * (self.TRAINING_FILES.shape[0] - 1))
        # self.current_files = self.TRAINING_FILES[rand_num]
        self.file_num += 1
        if self.file_num >= len(self.TRAINING_FILES):
            self.file_num = 0
        self.current_files = self.TRAINING_FILES[self.file_num]
        # if not 'risoners' in self.current_files[1]:  # train on just prisioners
        #     self.file_num = self.load_next_trn_file()
        return self.file_num

    def change_proposals_to_experts(self, text: object) -> object:
        if 'risoner' in self.current_files[0]:
            text = re.sub('@ax', '@fair', text)
            text = re.sub('@ay', '@bully', text)
            text = re.sub('@bx', '@bullied', text)
            text = re.sub('@by', '@sad', text)
        elif 'hicken' in self.current_files[0]:
            text = re.sub('@by', '@fair', text)
            text = re.sub('@bx', '@bully', text)
            text = re.sub('@ay', '@bullied', text)
            text = re.sub('@ax', '@sad', text)
        elif 'locks' in self.current_files[0]:
            # text = re.sub('', ' @fair', text) the fair strategy is a combination of bully and bullied
            text = re.sub('@cx', '@bully', text)
            text = re.sub('@ay', '@bully', text)
            text = re.sub('@az', '@bullied', text)
            text = re.sub('@bx', '@bullied', text)
            text = re.sub('@ax', '@sad', text)
            text = re.sub('@by', '@sad', text)
            text = re.sub('@bz', '@sad', text)
            text = re.sub('@cy', '@sad', text)
            text = re.sub('@cz', '@sad', text)
        elif 'ndless' in self.current_files[0]:
            text = re.sub('@ay', '@fair', text)
            text = re.sub('@ax', '@bully', text)
            text = re.sub('@by', '@bullied', text)
            text = re.sub('@bx', '@sad', text)
        return text

    def create_artificial_text(self, texts):
        new_texts = []
        for text in texts:
            new_text = text.lower()
            new_text = re.sub('no messages sent', ' ', new_text)  # get ride of No Message in messages and make them blank.
            new_text = re.sub(' (?P<group1>[abc])-+(?P<group2>[xyz])', ' @\g<group1>\g<group2>', new_text)  # change a-x to @ax ...
            # change @ax and other game chats to @fair depending on the game being played
            new_text = self.change_proposals_to_experts(new_text)

            for i in range(self.possible_chats_expanded.shape[0]):
                pattern = self.possible_chats_expanded[i][0]
                replacement = 'None'
                while replacement == 'None':
                    replacement = self.possible_chats_expanded[i][random.randint(1,9)]
                new_text = re.sub(pattern.lower(), replacement.lower(), new_text)
            new_texts += [new_text]
        return np.array(new_texts)

    def get_messages_and_actions(self, format_text=True):
        data = pd.read_csv(self.current_files[0])
        try:
            speech_act_player_not_i = data['PlayerSpeechAct']
            speech_act_player_i = data['PartnerSpeechAct']
            action_player_not_i = data['PlayerAction']
            action_player_i = data['PartnerAction']
        except:
            speech_act_player_not_i = data[' PlayerSpeechAct ']
            speech_act_player_i = data[' PartnerSpeechAct ']
            action_player_not_i = data[' PlayerAction ']
            action_player_i = data[' PartnerAction ']

        speech_act_player_i[pd.isnull(speech_act_player_i)] = ' '
        speech_act_player_not_i[pd.isnull(speech_act_player_not_i)] = ' '

        speech_act_player_i = self.create_artificial_text(np.array(speech_act_player_i))
        speech_act_player_not_i = self.create_artificial_text(np.array(speech_act_player_not_i))
        if format_text:
            return [self.format_text(np.array(speech_act_player_i)), self.format_text(
                np.array(speech_act_player_not_i)), np.array(action_player_i), np.array(action_player_not_i), speech_act_player_i, speech_act_player_not_i]
        return [np.array(speech_act_player_i), np.array(speech_act_player_not_i),
                np.array(action_player_i), np.array(action_player_not_i), speech_act_player_i, speech_act_player_not_i]

    def load_next_tst_file(self):
        self.current_files = self.TRAINING_FILES[-1]

    # def get_state_i(self):
    #     data = np.array(pd.read_csv(self.current_files[2], header=None))
    #
    #     original_states = self.get_original_states(self.get_game())
    #     state_i = np.zeros((data.size, len(original_states)))
    #
    #     for i in range(len(data[0])):
    #         state = ' '.join(data[0, i].split())
    #         if state != '':
    #             state_i[i, original_states.index(state)] = 1
    #         else:
    #             state_i = state_i[:i, :]
    #             break
    #
    #     state_i = pd.DataFrame(state_i, columns=original_states)
    #     state_i = self.__probability_profile_to_state_representation(state_i)
    #
    #     state_i[:, 0] = (state_i[:, 0] / 2).astype('int').astype('float')
    #
    #     return state_i[::2]

    def sample_state(self, state):
        zeros = np.zeros(len(state))
        zeros[0] = state[0]
        state = state[1:]
        index = np.random.choice(np.arange(len(state)), p=state/np.sum(state))
        zeros[index + 1] = 1
        return zeros

    def get_state_i(self, reformed_state=True):
        data = np.array(pd.read_csv(self.current_files[2]))
        for i in range(len(data)):
            data[i] = self.sample_state(np.array(data)[i])
        if reformed_state:
            state_i = self.probability_profile_to_reformed_state_representation(data)
        else:
            state_i = self.__probability_profile_to_state_representation(data)
            state_i[:, 0] = (state_i[:, 0] / 4).astype('int').astype('float')

        # for i in range(len(state_i)):
        #     state_i[i] = self.sample_state(state_i[i])

        return state_i[1::4], data[1::4]

    def get_theta_not_i(self, reformed_state=True):
        if 'risoner' in self.current_files[1]:
            t = 0
            pass
        data = np.array(pd.read_csv(self.current_files[1]))
        if reformed_state:
            theta_not_i = self.probability_profile_to_reformed_state_representation(data)
        else:
            theta_not_i = self.__probability_profile_to_state_representation(data)
            theta_not_i[:, 0] = (theta_not_i[:, 0] / 4).astype('int').astype('float')

        return theta_not_i[::4], data[::4]


if __name__ == "__main__":
    # this is how to setup and use the Composite Sentences

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    added_vocab = ['@ax', '@ay', '@az', '@bx', '@by', '@bz', '@cx', '@cy', '@cz']
    tokenizer.add_tokens(added_vocab)

    training_data = TrainingData(30, tokenizer=tokenizer)
    Z_i, Z_ni, A_i, A_ni = training_data.get_messages_and_actions()
    print(Z_i, Z_ni)
    print(A_i, A_ni)
    state_i = training_data.get_state_i()
    print(state_i)
    theta_not_i = training_data.get_theta_not_i()
    print(theta_not_i)
    training_data.load_next_trn_file()

    # this shows how to convert back to text according to the vocabulary
    print(training_data.convert_to_words(Z_i[0, 0]))  # this line prints the first line of the
    print(training_data.convert_to_words(Z_ni[0, 0]))

