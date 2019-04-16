# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
from collections import Counter
import sklearn.feature_extraction
from sklearn.preprocessing import LabelEncoder
import re
import keras
from keras.models import load_model
import h5py
import nltk
from nltk import stem
from nltk.corpus import stopwords

nltk.download('stopwords')

from ae_coding import NLP_utils
from ae_coding.NLP_utils import *


# encode the training data into the format that a neural network model can recognize
# Args:
#   Train: a dataframe containing "REPORTED_TERM" which will be used to train a neural network
#
# Returns: a list being the input x of the neural network model, and the feature names which will be used to construct input x from test data
def encode_training_x_for_neural_net(Train):
    Train = Train.copy()
    # clean the reported terms
    Train['REPORTED_TERM'] = Train['REPORTED_TERM'].apply(lambda x: preprocess(x))

    # create bag-of-word matrix of intput training reported terms, and then convert to list
    Xtrain, nn_feature_names_ = create_BOW_sparse_mat(Train, 'REPORTED_TERM')
    Xtrain = Xtrain.todense().tolist()

    return Xtrain, nn_feature_names_


# encode the test data into the format that a neural network model can recognize
# Args:
#   Train: a dataframe containing "REPORTED_TERM" which will be used to train a neural network
#
# Returns: a list being the input x of the test data that can be fit into the neural network for prediction
def encode_test_x_for_neural_net(Test, feature_name):
    Test = Test.copy()  # create copy to avoid changing Test in outer scope
    # clean the reported terms
    Test['REPORTED_TERM'] = Test['REPORTED_TERM'].apply(lambda x: preprocess(x))

    # create bag-of-word matrix of intput training reported terms, and then convert to list
    Xtest, feature_names_ = create_BOW_sparse_mat(Test, 'REPORTED_TERM', feature_name)
    Xtest = Xtest.todense().tolist()

    return Xtest


# create a 5-layer neural network model with 2 dropout layers
# Args:
#   Xtrain: A sparse matrix of bag-of-words, this should be the output of apply create_BOW_sparse_mat()
#   dummy_y_train: One hot encoded dummy target variables, this should be the output of applying keras.utils.np_utils.to_categorical()
#   input_unit: an integer to indicate number of input units
#   hidden_layer_unit: an integer to indicate number of hidden layer units / the third layer units in this model
#   dropout_rate_1: a number from 0 to 1 indicating the dropout rate of the first droupout layer
#   dropout_rate_2: a number from 0 to 1 indicating the dropout rate of the second droupout layer
#
# Returns: a 5-layer neural network model - relu, dropout, relu, dropout, softmax - with customized input unit / hidden layer unit / dropout rates
def five_layer_neural_net(Xtrain, dummy_y_train, input_unit, hidden_layer_unit, dropout_rate_1=.1, dropout_rate_2=.1):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(input_unit, input_shape=(len(Xtrain[0]),), activation='relu'))
    model.add(keras.layers.core.Dropout(rate=dropout_rate_1))
    model.add(keras.layers.Dense(hidden_layer_unit, activation='relu'))
    model.add(keras.layers.core.Dropout(rate=dropout_rate_2))
    model.add(keras.layers.Dense(output_dim=dummy_y_train.shape[1], activation='softmax'))
    # compile modile
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model


# a function that trains a five layer neural network, with input relu layer, followed by a dropout layer, followed by a relu hidden layer, followed by a dropout layer, followed by an output softmax layer
# Args:
#   Train: a dataframe containing "REPORTED_TERM" and "PT_NAME" / "LLT_NAME" which will be used to train a neural network
#   Meddra_dict: the whole MedDRA dictionary containing column "PT_NAME" / "LLT_NAME" which will be used to construct a list of possible target values
#   target: String, name of target variable, i.e. the MedDRA term level that we are trying to predict - e.g. "PT_NAME" / "LLT_NAME"
#   input_unit: an integer to indicate number of input units
#   hidden_layer_unit: an integer to indicate number of hidden layer units / the third layer units in this model
#   dropout_rate_1: a number from 0 to 1 indicating the dropout rate of the first droupout layer
#   dropout_rate_2: a number from 0 to 1 indicating the dropout rate of the second droupout layer
#
# Returns:
#    1. a 5-layer neural network model
#    2. a list of feature names of the bag-of-word matrix which will be used to construct input x from test data
#    3. a list of all possible predicted terms will shall be used to decode the predicted target labels of the test data

def train_neural_net(Train, Meddra_dict, target, input_unit, hidden_layer_unit, dropout_rate_1=.1, dropout_rate_2=.1):
    # generate the input x and targets of the training data that can be fit into a neural network model
    # also store
    #     1. the feature names of the bag-of-word matrix that will be used to construct the input x of the test data
    #     2. all possible target values that will be used to decode the predicted y as all_possible_pt
    Xtrain, nn_feature_names = encode_training_x_for_neural_net(Train)
    dummy_y_train, all_possible_pt = encode_training_y_for_neural_net(Train, Meddra_dict, target)

    # initial the 5-layer neural network model
    model = five_layer_neural_net(Xtrain, dummy_y_train, 468, 268, dropout_rate_1=.1, dropout_rate_2=.1)

    # train the model with specified epoch numbers and batch size
    model.fit(np.array(Xtrain), dummy_y_train, epochs=50, batch_size=200)

    return model, nn_feature_names, all_possible_pt


# a function that trains a five layer neural network, with input relu layer, followed by a dropout layer, followed by a relu hidden layer, followed by a dropout layer, followed by an output softmax layer
# Args:
#   Test: a dataframe containing "REPORTED_TERM" and "PT_NAME" / "LLT_NAME" which will be used to test a neural network
#   model:
#   nn_feature_names:
#   Meddra_dict: the whole MedDRA dictionary containing column "PT_NAME" / "LLT_NAME" which will be used to construct a list of possible target values
#   target: String, name of target variable, i.e. the MedDRA term level that we are trying to predict - e.g. "PT_NAME" / "LLT_NAME"
#   input_unit: an integer to indicate number of input units
#   hidden_layer_unit: an integer to indicate number of hidden layer units / the third layer units in this model
#   dropout_rate_1: a number from 0 to 1 indicating the dropout rate of the first droupout layer
#   dropout_rate_2: a number from 0 to 1 indicating the dropout rate of the second droupout layer
#
# Output:
#   Return a dataframe with complete results
def pred_neural_net(Test, model, nn_feature_names, all_possible_pt, target):
    # encode the test reported terms
    Xtest = encode_test_x_for_neural_net(Test, nn_feature_names)

    # predict on the test data
    # pred = model.predict(Xtest)
    pred = model.predict(np.array(Xtest))
    output_df = pd.DataFrame(np.max(pred, axis=1).tolist())
    output_df.columns = ['CONF']
    pred = list(np.argmax(pred, axis=1))
    pred = [sorted(all_possible_pt, key=str.lower)[i] for i in pred]

    # convert the prediction into a dataframe as output of this function
    output_df[target + '_PRED'] = pred
    output_df["REPORTED_TERM"] = Test["REPORTED_TERM"].values
    return output_df





