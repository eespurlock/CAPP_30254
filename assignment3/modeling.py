'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 3: Update the Pipeline

PY file #3: creating and testing models
'''

#Need to import more from sklearn

#Imports
#Pandas and numpy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

#sklearn models
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,\
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier

#sklearn metrics
from sklearn.metrics import accuracy_score as accuracy,\
    log_loss, precision_score, recall_score, roc_auc_score

#Defined constants for this assignment
REGRESSION = "Logistic Regression"
KNN = "K Nearest Neighbors"
TREE = "Decision Trees"
SVM = "Support Vector Machines"
FOREST = "Random Forests"
EXTRA = "Extra Trees"
ADA_BOOSTING = "Ada Boosting"
BAGGING = "Bagging"

ACCURACY = "Accuracy"
AVG_PREC = "Average Precision"
BRIER = "Brier Score Loss"
LOG = "Log Loss"
PRECISION = "Precision"
RECALL = "Recall"
ROC_AUC = "ROC_AUC"

def split_by_date(df_all_data, split, variable, features):
    '''
    Splits the data by date listed in the split column

    Inputs:
        df_all_data: a pandas dataframe
        split: the name of the column we are splitting on

    Outputs:
        test_train_dict: a dictionary mapping a date range to a tuple of pandas
            dataframes with the training and testing data
    '''
    models_dict = {}
    time_series = df_all_data[split]
    final_date = time_series.max()

    #Initialize test and train dates
    end_train = time_series.min()
    begin_test = end_train
    end_test = begin_test

    while end_test < final_date:
        #The training data ends 180 days after than the end of the last train
        end_train = end_train + timedelta(days=180)
        begin_test = end_train + timedelta(days=1)
        end_test = begin_test + timedelta(days=180)
        dates = str(begin_test) + " - " + str(end_test)
        
        #Now we create the training and testing data
        train_filter = df_all_data[split] <= end_train
        train_data = df_all_data[train_filter]
        test_filter =\
            (df_all_data[split] <= end_test) &\
            (df_all_data[split] >= begin_test)
        test_data = df_all_data[test_filter]

        #Now we have to create the variable and features data
        train_variable = train_data[variable]
        train_features = train_data[features]
        test_variable = test_data[variable]
        test_features = test_data[features]

        #Now save all of this to the dictionary
        #By the end of this assignent, I suspect you will tell me I rely too
        #much on dictionaries
        models_dict[dates] = training_models(train_variable, train_features,\
            test_variable, test_features)

    return models_dict

def training_models(train_variable, train_features, test_variable,\
    test_features):
    '''
    Trains models on training data

    Inputs:
        

    Outputs:
        models_dict: a dictionary of models
    '''
    models_dict = {}
    models_dict[REGRESSION] = regression_modeling()
    models_dict[KNN] = knn_modeling()
    models_dict[FOREST], models_dict[EXTRA], models_dict[TREE] =\
        forest_modeling()
    models_dict[SVM] = svm_modeling()
    models_dict[ADA_BOOSTING] = ada_boost_modeling()
    models_dict[BAGGING] = bagging_modeling()
    
    for name, model_param in models_dict.items():
        for param, model_unfit in model_param.items():
            print(name, param)
            model = model_unfit.fit(train_features, train_variable)
            models_dict[name][param] = test_models(model, test_variable,\
                test_features)

    return models_dict

def regression_modeling():
    '''
    Creates multiple regression models

    Inputs:

    Outputs:
    '''
    reg_dict = {}
    C_VALS = [1.0, 1.2, 1.5, 2.0, 2.5, 5]
    for c in C_VALS:
        param = "Regulatization Strength: " + str(c)
        reg_dict[param] = LogisticRegression(C=c)
    return reg_dict

def knn_modeling():
    '''
    Creates multiple nearest neighbors models

    Inputs:

    Outputs:
    '''
    knn_dict = {}
    NEIGHBORS = [1, 2, 5, 10, 20, 50, 100]
    for k in NEIGHBORS:
        param = "K Neighbors: " + str(k)
        knn_dict[param] = KNeighborsClassifier(n_neighbors=k)
    return knn_dict

def svm_modeling():
    '''
    Creates multiple support vector machine models

    Inputs:

    Outputs:
    '''
    svm_dict = {}
    NU = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for n in NU:
        param = "Bounds on errors / support vectors: " + str(n)
        svm_dict[param] = NuSVC(nu=n, gamma='scale')
    return svm_dict

def forest_modeling():
    '''
    Creates multiple random forest models and multiple extra trees models
    (Random forests and extra trees take the same parameters, which is why
    we are putting them together. We will use the same depth for decision
    trees which is why this is with them)

    Inputs:

    Outputs:
    '''
    forest_dict = {}
    extra_dict = {}
    tree_dict = {}
    NUM_TREES = [5, 20, 50, 200]
    MAX_DEPTH = [1, 5, 10, 20, 50, 100, 200]
    MAX_LEAF_NODES = [10, 100, None]
    for depth in MAX_DEPTH:
        tree_param = "Max Depth of Trees: " + str(depth)
        tree_dict[tree_param] = DecisionTreeClassifier(max_depth=depth)
        for trees in NUM_TREES:
            for leaf in MAX_LEAF_NODES:
                param = "Number of Trees: " + str(trees) +\
                    ", Max Depth of Trees: " + str(depth) +\
                    ", Max Leaf Nodes: " + str(leaf)
                forest_dict[param] = RandomForestClassifier(n_estimators=trees,\
                    max_depth=depth, max_leaf_nodes=leaf)
                extra_dict[param] = ExtraTreesClassifier(n_estimators=trees,\
                    max_depth=depth, max_leaf_nodes=leaf)
    return forest_dict, extra_dict, tree_dict

def ada_boost_modeling():
    '''
    Creates multiple AdaBoost models

    Inputs:

    Outputs:
    '''
    ada_dict = {}
    N_ESTIMATORS = [10, 30, 50, 100, 200]
    LEARNING_RATE = [0.5, 1.0, 2.0]
    for n in N_ESTIMATORS:
        for rate in LEARNING_RATE:
            param = "Estimators: " + str(n) + ", Learning Rate: " + str(rate)
            ada_dict[param] = AdaBoostClassifier(n_estimators=n,\
                learning_rate=rate)
    return ada_dict

def bagging_modeling():
    '''
    Creates multiple bagging models

    Inputs:
        num_feat: the number of features in the training set

    Outputs:
    '''
    bag_dict = {}

    N_ESTIMATORS = [5, 10, 20, 30, 50]
    MAX_SAMPLES = [1, 10, 50, 100, 500]

    for n in N_ESTIMATORS:
        for sample in MAX_SAMPLES:
            param = "Estimators: " + str(n) + ", Samples: " + str(sample)
            bag_dict[param] = BaggingClassifier(n_estimators=n,\
                max_samples=sample)
    return bag_dict

def test_models(model, test_var, test_features):
    '''
    Tests and evaluates models on testing data

    Inputs:
        model: a trained model we want to test
        variable: column name of the variable
        features: list of column names of the features

    Outputs:
        eval_dict: a dictionary of model evaluations
    '''
    THRESHOLDS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    eval_dict = {}
    probabilities = model.predict_proba(test_features)[:,1]
    key = "No Threshold"
    roc_auc = roc_auc_score(y_true=test_var, y_score=probabilities)
    eval_dict[key] = {ROC_AUC: roc_auc}
    for thresh in THRESHOLDS:    
        calc_threshold = lambda x,y: 0 if x < y else 1
        predicted = np.array([calc_threshold(score, thresh) for score in
            probabilities])
        key = "Threshold: " + str(thresh)
        eval_dict[key] = evaluate_models(test_var, predicted)
    return eval_dict

def evaluate_models(true, predicted):
    '''
    Evaluates models on multiple evaluations metrics

    Inouts:

    Outputs:
    '''
    eval_dict = {}
    eval_dict[ACCURACY] = accuracy(y_true=true, y_pred=predicted)
    eval_dict[LOG] = log_loss(y_true=true, y_pred=predicted)
    eval_dict[PRECISION] = precision_score(y_true=true, y_pred=predicted)
    eval_dict[RECALL] = recall_score(y_true=true, y_pred=predicted)
    return eval_dict
