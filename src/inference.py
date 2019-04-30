import os
import numpy as np
import pandas as pd
import keras
from keras.engine.saving import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from config.constants import *
from config.path import model_path


class Inference:
    
    def __init__(self, tokenizer, class_idx, model_json_path=None, model_weights_path=None):
        self.tokenizer = tokenizer
        self.class_idx = class_idx
        self._load_model(model_json_path, model_weights_path)
            

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

    def prepare_text(self, input_text):
        """
        Pre-processes the input text into a sequence matrix
        """
        if not self.tokenizer:
            self.tokenizer = pickle.load(open(TOKENIZER, "rb"))
        
        input_text = pd.Series(input_text)
        sequences = self.tokenizer.texts_to_sequences(input_text)
        sequences_matrix = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='pre')

        return sequences_matrix
    
    
    def predict(self, input_sequence):
        """
        Predicts the PT term for the REPORTED_TERM passed and returns the PT_ID and PT_TERM
            :param self: 
            :param input_sequence: 
        """   
        output = np.argmax(self.model.predict(input_sequence))
        #med_dict = self.tokenizer.reader.int_to_pt
        return output, self.class_idx[output]
        #return (output, med_dict[output])