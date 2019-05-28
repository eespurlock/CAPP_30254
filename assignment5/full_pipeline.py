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
csv_name = 'projects_2012_2013.csv'

def pipeline(csv=csv_name, i=60, num_splits=3, split='date_posted'):
    '''
    Goes from the beginning to the end of the machine learning pipeline

    Inputs:
        csv_name: the pathway to a CSV file that has the data we want
            (this is initialized to the CSV file we were given for this
            assignment)
        i: the number of days we want something to be funded in
        num_splits: the number of training and testing splits
        split: the name of the column with dates we want to split on

    Outputs:
        models_eval: a pandas dataframe of the different models we have tested,
            the different parameters we have tried on them and the evaluation
            metrics we have used
    '''
    df_all_data = prep_data.import_data(csv)
    if df_all_data is None:
        return None
    all_cols = df_all_data.columns

    descriptions = prep_data.explore_data(df_all_data, all_cols)

    models_dict = modeling.split_by_date(df_all_data, split,\
        i, num_splits)
    
    return table_models_eval(models_dict)

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
    df_lst = []

    for dates, model_dict in models_eval.items():
        for model, param_dict in model_dict.items():
            for param, eval_dict in param_dict.items():
                for threshold, eval_outcome_dict in eval_dict.items():
                    for eval_name, outcome in eval_outcome_dict.items():
                        this_lst = [dates, model, param, eval_name, threshold,\
                            outcome]
                        df_lst.append(this_lst)

    df_evaluated_models = pd.DataFrame(np.array(df_lst), columns=col_lst)

    df_evaluated_models.to_csv("Modeling_Projects_2012_2013.csv")
    return df_evaluated_models
