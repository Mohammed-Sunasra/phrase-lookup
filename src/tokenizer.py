import pickle
import numpy as np
np.random.seed(42)
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from config.constants import *
from config.path import glove_dir, TOKENIZER

class Tokenize:
    
    def __init__(self, data_reader, max_words, max_len, test_size=0.20, shuffle=True, random_state=42):
        self.reader = data_reader
        self.data = data_reader.data
        self.X, self.y = self.data[INPUT_COL_NAME], self.data[OUTPUT_COL_NAME]
        self.max_words = max_words
        self.max_len = max_len
        self.tok, self.X, self.y = self._preprocess()
        self.pretrained_embeddings = self._load_word_vectors()
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y,
            test_size=test_size, shuffle=shuffle,
            random_state=random_state)

    def _preprocess(self):
        tok = Tokenizer(num_words=self.max_words)
        tok.fit_on_texts(self.X)
        sequences = tok.texts_to_sequences(self.X)
        sequence_matrix = pad_sequences(sequences, maxlen=self.max_len)
        y_encoded = np.array(pd.get_dummies(self.y.values))
        return tok, sequence_matrix, y_encoded

    def _load_word_vectors(self):
        embeddings_index = {}
        f = open(glove_dir/'glove.6B.200d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        
        embedding_matrix = np.zeros((len(self.tok.word_index) + 1, EMBEDDING_DIM))
        for word, i in self.tok.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def save_tokenizer(self, filepath):
        with open(str(filepath), 'wb') as pickle_file:
            pickle.dump(self.tok, pickle_file)

    
    def prepare_text(self, input_text):
        """
        This function pre-processes text and converts it into sequence matrix
        """
        if not self.tok:
            self.tok = pickle.load(open(TOKENIZER, "rb"))
            #self.max_len = max_len
        input_text = pd.Series(input_text)
        sequences = self.tok.texts_to_sequences(input_text)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=self.max_len, padding='pre')

        return sequences_matrix