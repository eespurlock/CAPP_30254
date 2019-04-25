'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 3: Update the Pipeline

PY file #1: putting the pipeline together
'''

#Imports
import pandas as pd
import numpy as np
import prep_data
import modeling

#Constant for this assignment
csv_file = 'projects_2012_2013.csv'

def pipeline(csv_name=csv_file):
    '''
    Goes from the beginning to the end of the machine learning pipeline

    Inputs:
        csv_name: the pathway to a CSV file that has the data we want
            (this is initialized to the CSV file we were given for this
            assignment)

    Outputs:
        models_eval: a dictionary of the different models we have tested, the
            different parameters we have tried on them and the evaluation
            metrics we have used
    '''

    df_all_data = prep_data.import_data(csv_name)
    if df_all_data is None:
        return None
    all_cols = df_all_data.columns

    #Now I need to explore the data
    #Then I need to clean up the data
    #Wait for feedback before I dig into this

    #TBD = prep_data.explore_data(df_all_data, all_cols)
    #df_all_data = prep_data.clean_data(df_all_data, all_cols)

    #I also need to generate the features in this next function
    df_all_data, variable, features, split = prep_data.generate_var_feat(
        df_all_data, all_cols)

    test_train_dict = modeling.split_by_date(df_all_data, split)
    models_eval = loop_through_dates(test_train_dict)
    
    return table_models_eval(models_eval)

def loop_through_dates(test_train_dict):
    '''
    Loops through all the dates for testing and training so we can create models
    on all of them and evaluate them

    Inputs:
        test_train_dict: a dictionary with the training and testing data

    Outputs:
        models_eval: a dictionary with the different models and their evaluation
    '''
    models_eval = {}
    for dates, data in test_train_dict.items():
        train_variable, train_features, test_variable, test_features = data
        models_eval[dates] = modeling.training_models(train_variable,\
        	train_features, test_variable, test_features)
    return models_eval

def table_models_eval(models_eval):
    '''
    Loops through the dictionary of models we have created 
    and puts those results into a pandas dataframe

    Inputs:
        models_eval: all the models we have created and the evaluation for them
    Output:
        df_evaluated_models: a dataframe listing the models, their evaluation
            metric and how well those models did on that metric
    '''
    col_lst = ['Date', 'Model Name', 'Parameters', 'Evaluation Name',
        'Threshold', 'Result']
    model_lst = []
    param_lst = []
    date_lst = []
    eval_lst = []
    thres_lst = []
    out_lst = []

    for dates, model_dict in models_eval.items():
        date_lst.append(dates)
        for model, param_dict in model_dict.items():
            model_lst.append(model)
            for param, eval_dict in param_dict.items():
                param_lst.append(param)
                for threshold, eval_outcome_dict in eval_dict.items():
                    thres_lst.append(threshold)
                    for eval_name, outcome in eval_outcome_dict.items():
                        eval_lst.append(eval_name)
                        out_lst.append(outcome)

    df_evaluated_models = pd.Dataframe(np.array(date_lst, model_lst, param_lst, 
        eval_dict, thres_lst, out_lst), columns=col_lst)

    return df_evaluated_models
