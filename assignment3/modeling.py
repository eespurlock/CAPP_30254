'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 3: Update the Pipeline

PY file #3: creating and testing models
'''

#Need to import more from sklearn

#Imports
import pandas as pd
from sklearn.cross_validation import train_test_split
import os.path
import numpy as np
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as accuracy
from datetime import datetime, timedelta

#Defined constants for this assignment
REGRESSION = "Logistic Regression"
KNN = "K Nearest Neighbors"
TREE = "Decision Trees"
SVM = "Support Vector Machines"
FOREST = "Random Forests"
BOOSTING = "Boosting"
BAGGING = "Bagging"
ACCURACY = "Accuracy"

MODELS_LST = [REGRESSION, KNN, TREE, SVM, FOREST, BOOSTING, BAGGING]

REG_PARAM = []
KNN_PARAM = [1, 2, 5, 10, 20, 50, 100]
TREE_PARAM = [1, 5, 10, 20, 50, 100, 200]
SVM_PARAM = []
FOREST_PARAM = []
BOOST_PARAM = []
BAG_PARAM = []

PARAM_DICT = {REGRESSION: REG_PARAM, KNN: KNN_PARAM, TREE: TREE_PARAM,
    SVM: SVM_PARAM, FOREST: FOREST_PARAM, BOOSTING: BOOST_PARAM,
    BAGGING: BAG_PARAM}

#I need to look this up
AUC_ROC = TBD

THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, AUC_ROC]

#Need to figure out what different evaluations I'll be using
EVALUATIONS = [ACCURACY]

def split_by_date(df_all_data, split):
    '''
    Splits the data by date listed in the split column

    Inputs:
        df_all_data: a pandas dataframe
        split: the name of the column we are splitting on

    Outputs:
        test_train_dict: a dictionary mapping a date range to a tuple of pandas
            dataframes with the training and testing data
    '''
    test_train_dict = {}

def training_models(train, variable, features):
    '''
    Trains models on training data

    Inputs:
        train: a pandas dataframe with training data
        variable: column name of the variable
        features: list of column names of the features

    Outputs:
        models_dict: a dictionary of models
    '''
    models_dict = {}
    for model in MODELS_LST:
        models_dict[model] = {}
        parameters = PARAM_DICT[model]
        for param in parameters:
            models_dict[model][param] = {}
            #Now I need to find a way to actually train the diffeent models

def test_eval_models(test, model, variable, features):
    '''
    Tests and evaluates models on testing data

    Inputs:
        test: a pandas dataframe with testing data
        model: a trained model we want to test
        variable: column name of the variable
        features: list of column names of the features

    Outputs:
        eval_dict: a dictionary of model evaluations
    '''
    eval_dict = {}
    for this_eval in EVALUATIONS:
        for thresh in THRESHOLDS:
            #Now I need to find a way to actually evaluate the models
