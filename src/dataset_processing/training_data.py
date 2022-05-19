import numpy as np
from src.dataset_processing.training_data_wikipedia import TrainingData as TrainingDataWikipedia
from src.dataset_processing.training_data_movie import TrainingData as TrainingDataMovie
from src.dataset_processing.training_data_ssharp import TrainingData as TrainingDataSsharp
from transformers import BertTokenizer
import pandas as pd
import random
import re


class TrainingData:
    def __init__(self, batch_size, word_buffer_len, dataset_names):
        self.batch_size = batch_size
        self.word_buffer_len = word_buffer_len
        self.dataset_names = dataset_names
        self.tdw, self.tdm, self.tds = None, None, None

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.added_vocab = ['@ax', '@ay', '@az', '@bx', '@by', '@bz', '@cx', '@cy', '@cz', '@fair', '@bully', '@bullied', '@sad']
        self.tokenizer.add_tokens(self.added_vocab)

        for name in self.dataset_names:
            if name == 'wikipedia':
                self.tdw = TrainingDataWikipedia(word_buffer_len, self.tokenizer)
            elif name == 'movie':
                self.tdm = TrainingDataMovie(batch_size)
            elif name == 'ssharp' or name == 'ssharp_chitchat':
                self.tds = TrainingDataSsharp(word_buffer_len, self.tokenizer)
            else:
                raise Exception("not a valid dataset name")

    def load_next_batch(self, dataset_names=None):
        if dataset_names == None:
            dataset_names = self.dataset_names
        elif isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        for name in dataset_names:
            if name == 'wikipedia':
                pass
                # self.tdw.load_next_sentence()
            elif name == 'movie':
                self.tdm.load_next_trn_file()
                pass
            elif name == 'ssharp' or name == 'ssharp_chitchat':
                self.tds.load_next_trn_file()
            else:
                raise Exception('invalid dataset name ' + name)

    def get_masked_input(self, zs_out):
        zs_inp = np.zeros((self.batch_size, self.word_buffer_len))
        mask = np.zeros((self.batch_size, self.word_buffer_len))
        for j in range(self.batch_size):
            zs_inp[j] = zs_out[j]
            for _ in range(np.random.randint(0, 15)):
                rand_int = np.random.randint(0, self.word_buffer_len - 1)
                zs_inp[j, rand_int] = 103  # 103 is [MASK]
                mask[j, rand_int] = 1

        # shift output to the left
        zs_out = np.concatenate((zs_out, np.zeros((self.batch_size, 1))), axis=1)
        return zs_inp, zs_out, mask

    @staticmethod
    def contains_proposal(param):
        # print("YOU ARE USING THE CHITCHAT NETWORK!")
        for i in range(len(param)):
            if param[i] > 30521:
                return True
        return False

    @staticmethod
    def is_blank(param):
        return 101 < param[1] < 103

    @staticmethod
    def remove_padding(text):
        text = re.sub('\[PAD\]', '', text)
        text = re.sub('\[CLS\]', '', text)
        text = re.sub('\[SEP\]', '', text)
        text = re.sub('\[MASK\]', '', text)
        text = re.sub('\  +', ' ', text)
        return text

    def get_batch(self, dataset_name):
        if dataset_name == 'wikipedia':
            zs_out = np.zeros((self.batch_size, self.word_buffer_len))
            for j in range(self.batch_size):
                zs_out[j] = self.tdw.get_sentence()
                self.tdw.load_next_sentence()
            return zs_out

        if dataset_name == 'movie':
            Z_i, Z_ni, Tar = self.tdm.get_messages_and_actions()
            Z_i = Z_i[:, 0, :]
            Z_ni = Z_ni[:, 0, :]
            Tar = Tar[:, 0, :]

            Tar = np.concatenate((Tar, np.zeros([self.batch_size, 1])), axis=1)
            return Z_i, Z_ni, Tar

        if dataset_name == 'ssharp':
            Z_i, Z_ni, A_i, A_ni, speech_i, speech_not_i = self.tds.get_messages_and_actions()
            proposal_i = self.get_proposals_from_text(speech_i, self.tds.get_game())
            proposal_not_i = self.get_proposals_from_text(speech_not_i, self.tds.get_game())
            proposal_i = self.shift_proposals(proposal_i)
            proposal_not_i = self.shift_proposals(proposal_not_i)
            State_i, state_i_org = self.tds.get_state_i()
            Theta_not_i, theta_ni_org = self.tds.get_theta_not_i()
            Theta_not_i, theta_ni_org = Theta_not_i[:-1], theta_ni_org[:-1]

            Z_i = np.concatenate((np.zeros([1, self.word_buffer_len]), Z_i[:, 0, :]), axis=0)
            Z_ni = np.concatenate((np.zeros([1, self.word_buffer_len]), Z_ni[:, 0, :]), axis=0)

            Tar = Z_i[1:self.batch_size + 1]
            Z_ni = Z_ni[:self.batch_size]
            Z_i = Z_i[:self.batch_size]
            A_i = np.concatenate(([-1], A_i))[:-1]
            A_ni = np.concatenate(([-1], A_ni))[:-1]
            A = np.concatenate((A_i.reshape((A_i.shape[0]), 1), A_ni.reshape((A_ni.shape[0]), 1)), axis=1)
            compiled_state = np.concatenate([A, State_i, Theta_not_i, proposal_i, proposal_not_i], axis=1)[:self.batch_size]
            # compiled_state = np.concatenate([A, State_i, Theta_not_i], axis=1)[:self.batch_size]  # I removed proposals because they don't seem to do anything
            Tar = np.concatenate((Tar, np.zeros([self.batch_size, 1])), axis=1)

            return Z_i, Z_ni, compiled_state, Tar, state_i_org, theta_ni_org

        if dataset_name == 'ssharp_chitchat':
            Z_i, Z_ni, compiled_state, Tar, state_i_org, theta_ni_org = self.get_batch('ssharp')
            i = 0
            while i < Tar.shape[0]:
                if self.contains_proposal(Tar[i]) or self.is_blank(Tar[i]):
                    Z_i = np.append(Z_i[:i], Z_i[i+1:], axis=0)
                    Z_ni = np.append(Z_ni[:i], Z_ni[i + 1:], axis=0)
                    compiled_state = np.append(compiled_state[:i], compiled_state[i + 1:], axis=0)
                    Tar = np.append(Tar[:i], Tar[i + 1:], axis=0)
                    state_i_org = np.append(state_i_org[:i], state_i_org[i + 1:], axis=0)
                    theta_ni_org = np.append(theta_ni_org[:i], theta_ni_org[i + 1:], axis=0)
                else:
                    i += 1
            if i == 0:
                self.load_next_batch('ssharp_chitchat')
                return self.get_batch('ssharp_chitchat')
            num = int(self.batch_size / i + 1)
            Z_i, Z_ni, compiled_state, Tar, state_i_org, theta_ni_org = \
                np.tile(Z_i, [num, 1])[:self.batch_size], np.tile(Z_ni, [num, 1])[:self.batch_size], \
                np.tile(compiled_state, [num, 1])[:self.batch_size], np.tile(Tar, [num, 1])[:self.batch_size], \
                np.tile(state_i_org, [num, 1])[:self.batch_size], np.tile(theta_ni_org, [num, 1])[:self.batch_size]
            return Z_i, Z_ni, compiled_state, Tar, state_i_org, theta_ni_org

    def change_experts_to_proposals(self, text: object, game) -> object:
        if 'risoner' in game:
            text = re.sub('@fair', 'ax', text)
            text = re.sub('@bully', 'ay', text)
            text = re.sub('@bullied', 'bx', text)
            text = re.sub('@sad', 'by', text)
        elif 'hickenalt' in game:
            text = re.sub('@fair', 'ax', text)
            text = re.sub('@bully', 'ay', text)
            text = re.sub('@bullied', 'bx', text)
            text = re.sub('@sad', 'ay', text)
        elif 'hicken' in game:
            text = re.sub('@fair', 'by', text)
            text = re.sub('@bully', 'bx', text)
            text = re.sub('@bullied', 'ay', text)
            text = re.sub('@sad', 'ax', text)
        elif 'locks' in game:
            text = re.sub('@fair', 'cx-az', text)
            text = re.sub('@bully', 'cx', text)
            text = re.sub('@bullied', 'az', text)
            text = re.sub('@sad', 'ax', text)
        elif 'ndless' in game:
            text = re.sub('@fair', 'ay', text)
            text = re.sub('@bully', 'ax', text)
            text = re.sub('@bullied', 'by', text)
            text = re.sub('@sad', 'bx', text)
        return text

    def convert_to_words(self, text, game):
        text = self.tokenizer.decode(text)
        text = re.sub(' ’ ', '\'', text)
        text = re.sub(' \' ', '\'', text)
        return self.change_experts_to_proposals(text, game)

    def change_proposals_to_experts(self, text: object, game) -> object:
        if 'risoner' in game:
            text = re.sub('@ax', '@fair', text)
            text = re.sub('@ay', '@bully', text)
            text = re.sub('@bx', '@bullied', text)
            text = re.sub('@by', '@sad', text)
        elif 'hickenalt' in game:
            text = re.sub('@ax', '@fair', text)
            text = re.sub('@ay', '@bully', text)
            text = re.sub('@bx', '@bullied', text)
            text = re.sub('@by', '@sad', text)
        elif 'hicken' in game:
            text = re.sub('@by', '@fair', text)
            text = re.sub('@bx', '@bully', text)
            text = re.sub('@ay', '@bullied', text)
            text = re.sub('@ax', '@sad', text)
        elif 'locks' in game:
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
        elif 'ndless' in game:
            text = re.sub('@ay', '@fair', text)
            text = re.sub('@ax', '@bully', text)
            text = re.sub('@by', '@bullied', text)
            text = re.sub('@bx', '@sad', text)
        return text

    def create_artificial_text(self, texts, game):
        new_texts = []
        for text in texts:
            new_text = text.lower()
            new_text = re.sub(' (?P<group1>[abc])(?P<group2>[xyz])', ' @\g<group1>\g<group2>', new_text)  # change ax to @ax ...
            new_text = re.sub(' (?P<group2>[xyz])(?P<group1>[abc])', ' @\g<group1>\g<group2>', new_text)  # change xa to @ax ...
            new_text = re.sub(' (?P<group1>[abc])-+(?P<group2>[xyz])', ' @\g<group1>\g<group2>', new_text)  # change a-x to @ax ...
            new_text = re.sub(' (?P<group2>[xyz])-+(?P<group1>[abc])', ' @\g<group1>\g<group2>', new_text)  # change x-a to @ax ...
            # change @ax and other game chats to @fair depending on the game being played
            new_text = self.change_proposals_to_experts(new_text, game)
            new_texts += [new_text]

        return np.array(new_texts)

    def get_proposals_from_text(self, texts, game):
        texts = self.create_artificial_text(texts, game)
        proposals = []
        for text in texts:
            proposal = np.zeros(4)
            if '@fair' in text:
                proposal[0] = 1
            if '@bully' in text:
                proposal[1] = 1
            if '@bullied' in text:
                proposal[2] = 1
            if '@sad' in text:
                proposal[3] = 1
            proposals += [proposal]
        return np.array(proposals)

    @staticmethod
    def shift_proposals(texts):
        return np.append(np.zeros((1, 4)), texts[:-1], axis=0)

    def convert_to_tokens(self, text):
        trn_data = np.zeros((1, self.word_buffer_len))
        if isinstance(text, str):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = self.tokenizer.encode(text[0])
        for j, token in enumerate(tokens[:self.word_buffer_len]):
            trn_data[0, j] = token
        return np.array(trn_data).reshape((1, self.word_buffer_len))

    def get_vocabulary(self):
        return list(self.tokenizer.vocab) + list(self.tokenizer.get_added_vocab())
