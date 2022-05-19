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

        # global variables
        self.word_buffer_len = word_buffer_len

        # autoencoder variables
        self.articles_path = DATASET_PATH + 'wikipedia/en/articles/'
        self.articles = os.listdir(self.articles_path)
        self.current_article_id = 0
        self.NUM_ARTICLES = len(self.articles)
        self.current_article = self.get_new_article()
        self.current_sentence_id = -1
        self.num_sentences = -1
        self.current_sentences = self.get_article_text()

    def get_new_article(self):
        self.current_article_id += 1
        if self.current_article_id >= self.NUM_ARTICLES:
            self.current_article_id = 0
        self.current_article = self.articles[self.current_article_id]
        self.get_article_text()
        return self.current_article

    def get_article(self):
        return self.current_article

    def get_article_text(self):
        sentences = []
        for text in open(self.articles_path + self.current_article, encoding="utf8").read().lower().split('\n'):
            sentence = np.zeros(self.word_buffer_len)
            if np.random.randint(0, 10) == 0:  # This randomly removes periods from the end of the sentence
                tokens = self.tokenizer.encode(text[:-1])
            else:
                tokens = self.tokenizer.encode(text)

            for j, token in enumerate(tokens[:self.word_buffer_len]):
                sentence[j] = token

            sentences += [sentence]
            self.sentences = np.array(sentences)

        self.current_sentence_id = 0
        self.num_sentences = len(self.sentences)
        self.current_sentences = self.sentences
        return self.sentences

    def get_sentence(self):
        return np.array(self.current_sentences[self.current_sentence_id])

    def load_next_sentence(self):
        self.current_sentence_id += 1
        # check if there are still sentences
        if self.current_sentence_id >= self.num_sentences:
            self.current_sentence_id = 0
            self.num_sentences = 0
            self.get_new_article()
            self.get_article_text()


if __name__ == "__main__":
    # Here is an example of how to use each function in the training data class
    # this is to set up the autoencoder
    training_data = TrainingData(40)
    training_data.load_next_sentence()
    training_data.load_next_sentence()
    training_data.load_next_sentence()
    training_data.load_next_sentence()
    text = training_data.get_sentence()
    print(text)

    # this shows how to convert back to text according to the vocabulary
    print(training_data.convert_to_words(text))

