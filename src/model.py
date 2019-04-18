import os
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout, SpatialDropout1D, LSTM
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from config.constants import *
from config.path import model_path


class LSTMModel:

    def __init__(self, tokenizer, loss=None, optimizer=None, metrics=None, batch_size=64, epochs=50, shuffle=True):
        self.tokenizer = tokenizer
        self.max_words = len(self.tokenizer.tok.word_index) + 1
        self.input_shape = self.tokenizer.X_train.shape[1]
        self.no_of_classes = self.tokenizer.y.shape[1]
        self.model = self._create_rnn(self.tokenizer.pretrained_embeddings)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle

    def _create_rnn(self, pretrained_embeddings=None):
        model = Sequential()
        model.add(Embedding(self.max_words, EMBEDDING_DIM, input_length=self.input_shape, weights=[pretrained_embeddings]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(self.no_of_classes, activation='softmax'))
        return model
    
    def fit(self):
        self.model.fit(self.tokenizer.X_train, self.tokenizer.y_train,
                       batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=[self.tokenizer.X_val, self.tokenizer.y_val],
                       shuffle=self.shuffle,
                       callbacks=[
                            ReduceLROnPlateau(),
                            EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001),
                            ModelCheckpoint(filepath=os.path.join(str(model_path) + "/", 'model_lstm_best_weights.h5'), save_best_only=True)])
    
    def predict(self, input_sequence):
        output = self.model.predict(input_sequence)
        med_dict = self.tokenizer.reader.pt_to_int
        return {output, med_dict[output]}

    def save(self, model_json_path, model_weights_path):
        with open(str(model_json_path), 'w') as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(str(model_weights_path))
