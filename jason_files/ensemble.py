# -*- coding: utf-8 -*-
"""

"""

from sklearn import linear_model
import numpy as np
import pandas as pd
import joblib


# A function that creates new features as input for the ensemble model
#
# Create 0/1 variables for whether predicted PT names from one model agrees with PT predictions from each of the other models
# and create interaction terms between these variables and the confidence scores of each model
# Args:
#    modeldat: a dataframe with results from each of the previous model (including fuzzy matching on llt, fuzzy matching on pt, devision tree, and etc.)
#    ALL_MODEL_NAME: a list of model names. For example, ALL_MODEL_NAME = ['FUZZY_LLT', 'FUZZY_PT', 'NN']
#
# Returns:
#     A dataframe with new features as new columns in the input modeldat columns, and a list of variable names that will be used in the ensemble method
def create_ensemble_model_new_features(modeldat, ALL_MODEL_NAME):
    # create a list to store all variable names needed for ensemble model
    if 'PT_NAME_COMPL' in modeldat.columns.values:
        feature_name_ls = ['REPORTED_TERM', 'PT_NAME_COMPL'] + ['PT_NAME_PRED_' + x for x in ALL_MODEL_NAME]
    else:
        feature_name_ls = ['REPORTED_TERM'] + ['PT_NAME_PRED_' + x for x in ALL_MODEL_NAME]

        # create a copy of the input modeldat
    modeldat = modeldat.copy()

    # create input features for each linear regression model, each regression model app
    for one_model_name in ALL_MODEL_NAME:
        for modelname in set(ALL_MODEL_NAME) - set([one_model_name]):
            confvar = 'CONF_' + one_model_name
            agreesvar = 'AGREES_' + modelname
            modeldat[agreesvar] = (modeldat['PT_NAME_PRED_' + one_model_name].copy() == modeldat[
                'PT_NAME_PRED_' + modelname].copy()).astype(int)
            modeldat[confvar + '_x_' + agreesvar] = modeldat[confvar].astype(float).copy() * modeldat[agreesvar].astype(
                float).copy()
            feature_name_ls.append(agreesvar)
            feature_name_ls.append(confvar + '_x_' + agreesvar)

        # Create 0/1 target variable: is this model correct?
        if 'PT_NAME_COMPL' in modeldat.columns.values:
            modeldat['PROB_CORRECT_' + one_model_name] = (
                        modeldat['PT_NAME_COMPL'].copy() == modeldat['PT_NAME_PRED_' + one_model_name].copy()).astype(
                int)
        else:
            modeldat['PROB_CORRECT_' + one_model_name] = np.nan

        feature_name_ls.append('PROB_CORRECT_' + one_model_name)

    return modeldat, feature_name_ls


# A function that generates input x & target y for each linear regression in the ensemble model
#
# Args:
#    modeldat: a dataframe with results from each of the previous model (including fuzzy matching on llt, fuzzy matching on pt, devision tree, and etc.)
#    ALL_MODEL_NAME: a list of model names. For example, ALL_MODEL_NAME = ['FUZZY_LLT', 'FUZZY_PT', 'NN']
#    one_model_name: one string from list ALL_MODEL_NAME, e.g. 'FUZZY_LLT'
#
# Returns:
#     Two lists being x and y, which can be fit into the regression models

def build_input_feature_for_one_model(modeldat, ALL_MODEL_NAME, one_model_name):
    modeldat = modeldat.copy()

    conf_features = [x for x in modeldat.columns.values if ('CONF_' in x) and ('AGREES_' not in x)]
    interaction_features = [x for x in modeldat.columns.values if ('AGREES_' in x) and ('CONF_' + one_model_name in x)]

    varnames = sorted(conf_features + interaction_features)
    modeldat = modeldat.fillna(0)

    X = modeldat[varnames].values.tolist()
    Y = modeldat["PROB_CORRECT_" + one_model_name].values.tolist()

    return X, Y


# A function that trains each linear regression model in the ensemble and returns the linear regression model
#
# Args:
#    X: A list of sublists with numbers, as input x of the regression model
#    Y: A list of target y values
#
# Returns:
#     A trained regression model

def run_ensemble_individual_regression(X, Y):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X, Y)
    return regr


# A function that uses a pre-rained model for prediction
#
# Args:
#    X: A list of sublists with numbers, as input x to do prediction using the regression model
#    model: A pre-trained linear regression model
#
# Returns:
#    An array of predicted values
def pred_ensemble_individual_regression(X, model):
    # Make predictions using the testing set
    y_pred = model.predict(X)
   
    return y_pred


# A function that trains the ensemble model, i.e. multiple sub linear models, and return the final prediction of the ensemble model
#
# Args:
#    modeldat: a dataframe with results from each of the previous model (including fuzzy matching on llt, fuzzy matching on pt, devision tree, and etc.)
#    ALL_MODEL_NAME: a list of model names. For example, ALL_MODEL_NAME = ['FUZZY_LLT', 'FUZZY_PT', 'NN']
#    output_file_path: a path where to store the sub linear models
#
# Returns:
#     A data frame with combined predicted results from sub linear models and the finalized prediction results from the ensemble model
#     (note that sub linear regression models will be saved as seperated pickle files under output_file_path after running this function )
def train_ensemble_model(modeldat, ALL_MODEL_NAME, output_file_path):
    modeldat = modeldat.copy()

    modeldat, allvars = create_ensemble_model_new_features(modeldat, ALL_MODEL_NAME)

    modeldat = modeldat.fillna(0)

    df = modeldat[['PT_NAME_PRED_' + x for x in ALL_MODEL_NAME]]
    df['REPORTED_TERM'] = modeldat['REPORTED_TERM'].values

    for itr, each_model_name in enumerate(ALL_MODEL_NAME):
        x, y = build_input_feature_for_one_model(modeldat, ALL_MODEL_NAME, each_model_name)
        # run each linear regression model and save the model
        regr_model = run_ensemble_individual_regression(x, y)
        with open(output_file_path + each_model_name + '_REG_MODEL.pkl', 'wb') as f:
            joblib.dump(regr_model, f)

        y_pred= pred_ensemble_individual_regression(x, regr_model)
        df['SCORE_' + each_model_name] = y_pred
    df['FINAL_PREDICTION'] = df[['SCORE_' + i for i in ALL_MODEL_NAME]].idxmax(axis=1).apply(
        lambda x: 'PT_NAME_PRED' + x[5:])
    # close the pickle file

    for itr, each_model_name in enumerate(ALL_MODEL_NAME):

        sub_df = df[df['FINAL_PREDICTION'] == 'PT_NAME_PRED_' + each_model_name]
        sub_df['FINAL_PRED_PT'] = sub_df['PT_NAME_PRED_' + each_model_name]
        if itr == 0:
            final_output = sub_df
        else:
            final_output = pd.concat([final_output, sub_df])

    return final_output


# A function that use the ensemble model for prediction
#
# Args:
#    modeldat: a dataframe with results from each of the previous model (including fuzzy matching on llt, fuzzy matching on pt, devision tree, and etc.)
#    ALL_MODEL_NAME: a list of model names. For example, ALL_MODEL_NAME = ['FUZZY_LLT', 'FUZZY_PT', 'NN']
#    model_path: a path stores the pre-trained sub linear models
#
# Returns:
#     A data frame with combined predicted results from sub linear models and the finalized prediction results from the ensemble model

def pred_ensemble_model(modeldat, ALL_MODEL_NAME, model_path):
    modeldat = modeldat.copy()

    modeldat, allvars = create_ensemble_model_new_features(modeldat, ALL_MODEL_NAME)

    for var in allvars:
        modeldat[var] = modeldat[var].fillna(0)

    df = modeldat.copy()

    for itr, each_model_name in enumerate(ALL_MODEL_NAME):
        x, y = build_input_feature_for_one_model(modeldat, ALL_MODEL_NAME, each_model_name)
        #print('X', x)
        # read each pre-trained linear regression model
        regr_model = joblib.load(model_path + each_model_name + '_REG_MODEL.pkl')
        y_pred= pred_ensemble_individual_regression(x, regr_model)
        
        df['SCORE_' + each_model_name] = y_pred
    df['FINAL_PREDICTION'] = df[['SCORE_' + i for i in ALL_MODEL_NAME]].idxmax(axis=1).apply(
        lambda x: 'PT_NAME_PRED' + x[5:])
    
    temp=df[['CONF_' + i for i in ALL_MODEL_NAME]]
    p=np.tile(np.array(temp.columns), temp.shape[0])
    for x, y in zip(temp.iterrows(),p[np.argsort(-temp)]): 
        a = x[1][y]
        for i in range(5):
            df.loc[x[0],'FINAL_PREDICTION_'+str(i+2)] = df.loc[x[0],'PT_NAME_PRED_'+a.index[i].split('CONF_')[1]]
            df.loc[x[0],'CONF_FINAL_PREDICTION_'+str(i+2)] = a[i]
        
     
    # close the pickle file

    for itr, each_model_name in enumerate(ALL_MODEL_NAME):

        sub_df = df[df['FINAL_PREDICTION'] == 'PT_NAME_PRED_' + each_model_name].copy()
        
        sub_df['FINAL_PRED_PT'] = sub_df['PT_NAME_PRED_' + each_model_name].values
        #sub_df['PREDICTIONTOP1'] = sub_df['PREDICTIONTOP1' +each_model_name ].values
        if itr == 0:
            final_output = sub_df
        else:
            final_output = pd.concat([final_output, sub_df])
    
    #final_output = final_output.rename(columns={'TOP2': 'FINAL_PREDICTION_2', 'TOP3': 'FINAL_PREDICTION_3','TOP4': 'FINAL_PREDICTION_4', 'TOP5': 'FINAL_PREDICTION_5','TOP6': 'FINAL_PREDICTION_6',
     #                                           'CONF_TOP2':'CONF_FINAL_PREDICTION_2', 'CONF_TOP3':'CONF_FINAL_PREDICTION_3', 'CONF_TOP4':'CONF_FINAL_PREDICTION_4', 'CONF_TOP5':'CONF_FINAL_PREDICTION_5','CONF_TOP6':'CONF_FINAL_PREDICTION_6'})
    final_output = final_output[['REPORTED_TERM', 'FINAL_PRED_PT','FINAL_PREDICTION_2','CONF_FINAL_PREDICTION_2','FINAL_PREDICTION_3','CONF_FINAL_PREDICTION_3','FINAL_PREDICTION_4','CONF_FINAL_PREDICTION_4','FINAL_PREDICTION_5','CONF_FINAL_PREDICTION_5','FINAL_PREDICTION_6','CONF_FINAL_PREDICTION_6']]
    #final_output["exists"] = final_output.eq(final_output.pop('FINAL_PRED_PT'), axis=0).any(axis=1)
    return final_output

