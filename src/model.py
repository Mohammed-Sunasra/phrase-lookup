import os
import numpy as np
import keras
from keras.engine.saving import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout, SpatialDropout1D, LSTM
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from config.constants import *
from config.path import model_path


class LSTMModel:
    
    def __init__(self, tokenizer, loss=None, optimizer=None, metrics=None, batch_size=None, epochs=None,
                model_json_path=None, model_weights_path=None, shuffle=True, eval=False):
        self.tokenizer = tokenizer
        if not eval:
            self.model = self._create_rnn(self.tokenizer.pretrained_embeddings)
            self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            self.batch_size = batch_size
            self.epochs = epochs
            self.shuffle = shuffle
        else:
            self._load_model(model_json_path, model_weights_path)
            

    def _create_rnn(self, pretrained_embeddings=None):
        """
        Creates an LSTM based model with pretrained Glove word embeddings
            :param self: 
            :param pretrained_embeddings=None: 
        """   
        model = Sequential()
        model.add(Embedding(self.tokenizer.max_words, EMBEDDING_DIM, input_length=self.tokenizer.max_len, weights=[pretrained_embeddings]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(self.tokenizer.no_of_classes, activation='softmax'))
        return model
    
    def fit(self):
        """
        Trains the model and saves the best model weights
            :param self: 
        """   
        self.model.fit(self.tokenizer.X_train, self.tokenizer.y_train,
                       batch_size=self.batch_size, epochs=self.epochs,
                       validation_data=[self.tokenizer.X_val, self.tokenizer.y_val],
                       shuffle=self.shuffle,
                       callbacks=[
                            ReduceLROnPlateau(),
                            EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001),
                            ModelCheckpoint(filepath=os.path.join(str(model_path) + "/", 'model_lstm_best_weights.h5'), save_best_only=True)])
    
    
    def save(self, model_json_path, model_weights_path):
        """
        Saves the model structure and weights file in 'model_files' folder
            :param self: 
            :param model_json_path: 
            :param model_weights_path: 
        """   
        with open(str(model_json_path), 'w') as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(str(model_weights_path))


    def _load_model(self, model_json_path, model_weights_path):
        """
        Loads a pretrained model from json file and weights
            :param self: 
            :param model_json_path: 
            :param model_weights_path: 
        """   
        with open(str(model_json_path)) as json_file:
            self.model = model_from_json(json_file.read())
        self.model.load_weights(str(model_weights_path))

    
    def predict(self, input_sequence):
        """
        Predicts the PT term for the REPORTED_TERM passed and returns the PT_ID and PT_TERM
            :param self: 
            :param input_sequence: 
        """   
        output = np.argmax(self.model.predict(input_sequence))
        med_dict = self.tokenizer.reader.int_to_pt
        return (output, med_dict[output])