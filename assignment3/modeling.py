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
    GradientBoostingClassifier, AdaBoostingClassifier, BaggingClassifier

#sklearn metrics
from sklearn.metrics import accuracy_score as accuracy,\
    balanced_accuracy_score as balanced,\
    log_loss, precision_score, recall_score, roc_auc_score

#Defined constants for this assignment
REGRESSION = "Logistic Regression"
KNN = "K Nearest Neighbors"
TREE = "Decision Trees"
SVM = "Support Vector Machines"
FOREST = "Random Forests"
EXTRA = "Extra Trees"
GRAD_BOOSTING = "Gradient Boosting"
ADA_BOOSTING = "Ada Boosting"
BAGGING = "Bagging"

ACCURACY = "Accuracy"
BAL_ACC = "Balanced Accuracy"
AVG_PREC = "Average Precision"
BRIER = "Brier Score Loss"
LOG = "Log Loss"
PRECISION = "Precision"
RECALL = "Recall"
ROC_AUC = "ROC_AUC"

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
    models_dict[KNN] = knn_modelig()
    models_dict[FOREST], models_dict[EXTRA], models_dict[TREE] =\
        forest_modeling()
    models_dict[SVM] = svm_modeling()
    models_dict[GRAD_BOOSTING] = grad_boost_modeling()
    models_dict[ADA_BOOSTING] = ada_boost_modeling()
    models_dict[BAGGING] = bagging_modeling(len(train_features.columns))
    
    for name, model_param in models_dict.items():
        for param, model_unfit in model_param.items():
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
    NUM_TREES = [5, 10, 20, 50, 100, 200]
    MAX_DEPTH = [1, 5, 10, 20, 50, 100, 200]
    MAX_LEAF_NODES = [10, 50, 100, 200, 300, None]
    for trees in NUM_TREES:
        for depth in MAX_DEPTH:
            for leaf in MAX_LEAF_NODES:
                param = "Number of Trees: " + str(trees) +\
                    ", Max Depth of Trees: " + str(depth) +\
                    ", Max Leaf Nodes: " + str(leaf)
                tree_param = "Max Depth of Trees: " + str(depth)
                forest_dict[param] = RandomForestClassifier(n_estimators=trees,\
                    max_depth=depth, max_leaf_nodes=leaf)
                extra_dict[param] = ExtraTreesClassifier(n_estimators=trees,\
                    max_depth=depth, max_leaf_nodes=leaf)
                tree_dict[tree_param] = DecisionTreeClassifier(max_depth=depth)
    return forest_dict, extra_dict, tree_dict

def ada_boost_modeling():
    '''
    Creates multiple AdaBoost models

    Inputs:

    Outputs:
    '''
    ada_dict = {}
    N_ESTIMATORS = [10, 30, 50, 100, 200]
    LEARNING_RATE = [0.5, 1.0, 1.5, 2.0, 2.5]
    for n in N_ESTIMATORS:
        for rate in LEARNING_RATE:
            param = "Estimators: " + str(n) + ", Learning Rate: " + str(rate)
            ada_dict[param] = AdaBoostingClassifier(n_estimators=n,\
                learning_rate=rate)
    return ada_dict

def grad_boost_modeling():
    '''
    Creates multiple Gradient Boosting models

    Inputs:

    Outputs:
    '''
    grad_dict = {}
    LEARNING_RATE = [0.1, 0.5, 1.0, 1.5, 2.0]
    N_ESTIMATORS = [20, 50, 100, 150, 200]
    SUBSAMPLE = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    MAX_DEPTH = [1, 5, 10, 20, 50, 100, 200]
    MAX_LEAF_NODES = [10, 50, 100, 200, 300, None]
    for r in LEARNING_RATE:
        for n in N_ESTIMATORS:
            for s in SUBSAMPLE:
                for d in MAX_DEPTH:
                    for l in MAX_LEAF_NODES:
                        param = "Learning Rate: " + str(r) + ", Estimators: " +\
                            str(n) + ", Subsample: " + str(s) + ", Depth: " +\
                            str(d) + ", Leaf Nodes: " + str(l)
                        grad_dict[param] = GradientBoostingClassifier(\
                            learning_rate=r, n_estimators=n, subsample=s,\
                            max_depth=d, max_leaf_nodes=l)

def bagging_modeling(num_feat):
    '''
    Creates multiple bagging models

    Inputs:
        num_feat: the number of features in the training set

    Outputs:
    '''
    bag_dict = {}

    N_ESTIMATORS = [5, 10, 20, 30, 50]
    MAX_SAMPLES = [1, 5, 10, 20, 50, 100, 500]
    MAX_FEATURES = [1, 2, 5, 10, 20] + [num_feat]

    for n in N_ESTIMATORS:
        for sample in MAX_SAMPLES:
            for feat in MAX_FEATURES:
                param = "Estimators: " + str(n) + ", Samples: " + str(sample)\
                    + ", Features: " + str(feat)
                bag_dict[param] = BaggingClassifier(n_estimators=n,\
                    max_samples=sample, max_features=feat)

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
        predicted = np.array([calc_threshold(score, threshold) for score in
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
    eval_dict[BAL_ACC] = balanced(y_true=true, y_pred=predicted, adjusted=True)
    eval_dict[LOG] = log_loss(y_true=true, y_pred=predicted)
    eval_dict[PRECISION] = precision_score(y_true=true, y_pred=predicted)
    eval_dict[RECALL] = recall_score(y_true=true, y_pred=predicted)
    return eval_dict
