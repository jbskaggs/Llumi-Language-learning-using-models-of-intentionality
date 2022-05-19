import os
import numpy as np
# import pyttsx3
import re
import csv
import pandas as pd
import json
from transformers import BertTokenizer

# engine = pyttsx3.init()
# engine.say("I will speak this text")
# engine.runAndWait()


def edit_text(text):
    text = re.sub('\n', '', text).lower()
    return text


edit_text_v = np.vectorize(edit_text)


class TrainingData:
    def __init__(self, batch_size):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.added_vocab = ['@ax', '@ay', '@az', '@bx', '@by', '@bz', '@cx', '@cy', '@cz']
        self.tokenizer.add_tokens(self.added_vocab)

        self.batch_size = batch_size
        MARVEL_MOVIE_DIALOG_PATH = '/home/jskaggs93/Datasets/MarvelMovieDialogs/'
        CORNELL_MOVIE_DIALOG_PATH = '/home/jskaggs93/Datasets/CornellMovieDialog/'
        # TEXT_DATASET_FOR_NLP_PATH = '/home/jskaggs93/Datasets/TextDataSetforNLP/'
        GUTENBERG_DATASET_PATH = '/home/jskaggs93/Datasets/gutenberg_dataset_en/'

        # global variables
        self.word_buffer_len = 40

        # Cornell Movie Dataset
        cornel_movie_texts = open(CORNELL_MOVIE_DIALOG_PATH + 'movie_lines.txt').readlines()
        line_id, character_name = [], []
        for i in range(len(cornel_movie_texts)):
            line_id += [int(cornel_movie_texts[i].split('+++$+++')[0][1:])]
            character_name += [cornel_movie_texts[i].split('+++$+++')[3]]
            cornel_movie_texts[i] = cornel_movie_texts[i].split('+++$+++')[4]

        cornel_movie_texts = np.array(cornel_movie_texts)
        cornel_movie_texts = edit_text_v(cornel_movie_texts)
        self.cornel_line_id, self.cornel_character_id, self.cornel_text = zip(*sorted(zip(line_id, character_name, cornel_movie_texts)))
        self.cornell_idx = 0

        # Marvel Dataset
        marvel_movies = os.listdir(MARVEL_MOVIE_DIALOG_PATH)
        self.marvel_movie_texts = []
        for movie in marvel_movies:
            if movie != '.DS_Store':
                tmp = open(MARVEL_MOVIE_DIALOG_PATH + movie, encoding="utf8")
                text = np.array(tmp.readlines())
                self.marvel_movie_texts += [edit_text_v(text)]
        self.marvel_current_movie_idx = 0
        self.marvel_current_movie_text_idx = 0

        # Gutenberg Dataset
        self.gutenberg_texts = open(GUTENBERG_DATASET_PATH + 'train.txt').readlines()
        self.gutenberg_idx = 0
        self.make_test_set()

    def make_test_set(self):
        self.test_z_i, self.test_z_ni, self.test_tar = [], [], []
        self.cornel_initial_id = 0
        for i in range(20):  # 800 examples
            self.load_next_cornell()
            self.test_z_i += self.z_i
            self.test_z_ni += self.z_ni
            self.test_tar += self.tar
        self.cornel_initial_id = self.cornell_idx

    def get_test_set(self):
        Z_i = self.format_text(self.test_z_i)
        Z_ni = self.format_text(self.test_z_ni)
        Tar = self.format_text(self.test_tar)
        return Z_i, Z_ni, Tar

    def load_next_trn_file(self):
        rand = np.random.randint(0, 200)
        # return self.load_next_gutenberg()
        if rand < 5:
            return self.load_next_cornell()
        else:
            return self.load_next_gutenberg()

    def load_next_marvel(self):
        if self.marvel_current_movie_text_idx + self.batch_size >= len(self.marvel_movie_texts[self.marvel_current_movie_idx]):
            self.marvel_current_movie_idx += 1
            self.marvel_current_movie_text_idx = 0
        if self.marvel_current_movie_idx >= len(self.marvel_movie_texts):
            self.marvel_current_movie_idx = 0
        self.text = self.marvel_movie_texts[self.marvel_current_movie_idx][self.marvel_current_movie_text_idx:self.marvel_current_movie_text_idx + self.batch_size]
        self.marvel_current_movie_text_idx += self.batch_size
        return self.text

    def load_next_cornell(self):
        z_i, z_ni, tar, i = [], [], [], 0
        while i < self.batch_size:
            if self.cornell_idx + 1 + i >= len(self.cornel_text):
                self.cornell_idx = self.cornel_initial_id
                print('restarting cornell dataset ...')
            if self.cornel_line_id[self.cornell_idx + i] == self.cornel_line_id[self.cornell_idx + i + 1] - 1:
                if self.cornell_idx + i > 0 and self.cornel_line_id[self.cornell_idx + i - 1] == self.cornel_line_id[self.cornell_idx + i + 1] - 2:
                    z_i += [self.cornel_text[self.cornell_idx + i - 1]]
                else:
                    z_i += [' ']
                z_ni += [self.cornel_text[self.cornell_idx + i]]
                tar += [self.cornel_text[self.cornell_idx + i + 1]]
                i += 1
            else:
                self.cornell_idx += 1
        self.cornell_idx += self.batch_size

        self.z_i, self.z_ni, self.tar = z_i, z_ni, tar
        return self.z_i, self.z_ni, self.tar

    def load_next_gutenberg(self):
        z_i, z_ni, tar, i = [], [], [], 0
        while i < self.batch_size:
            if self.gutenberg_idx + 1 + i >= len(self.gutenberg_texts):
                self.gutenberg_idx = 0
                print('restarting gutenberg dataset ...')
            if self.gutenberg_texts[self.gutenberg_idx + i] != '' and self.gutenberg_texts[self.gutenberg_idx + i + 1] != '':
                if self.gutenberg_idx + i > 0 and self.gutenberg_texts[self.gutenberg_idx + i - 1] != '':
                    z_i += [edit_text(self.gutenberg_texts[self.gutenberg_idx + i - 1])]
                else:
                    z_i += [' ']
                z_ni += [edit_text(self.gutenberg_texts[self.gutenberg_idx + i])]
                tar += [edit_text(self.gutenberg_texts[self.gutenberg_idx + i + 1])]
                i += 1
            else:
                self.gutenberg_idx += 1
        self.gutenberg_idx += self.batch_size

        self.z_i, self.z_ni, self.tar = z_i, z_ni, tar
        return self.z_i, self.z_ni, self.tar

    def get_messages_and_actions(self):
        Z_ni = self.format_text(self.z_ni)
        Z_i = self.format_text(self.z_i)
        Tar = self.format_text(self.tar)
        return Z_i, Z_ni, Tar

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

    def convert_to_words(self, text):
        return self.tokenizer.decode(text)


if __name__ == "__main__":
    td = TrainingData(20)
    td.load_next_trn_file()
    a, b, c = td.get_messages_and_actions()
    for i in range(20):
        print('Person A: ' + td.convert_to_words(a[i, 0, :]))
        print('Person B: ' + td.convert_to_words(b[i, 0, :]))
        print('Person A: ' + td.convert_to_words(c[i, 0, :]))
        print()

    # for s, t in zip(td.cornel_movie_speaker[:20], td.cornel_movie_texts[:20]):
    #     print(s + ': ' + t)
