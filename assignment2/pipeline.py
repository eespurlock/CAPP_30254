'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 2: Machine Learning Pipeline
'''

#Imports
import pandas as pd
import sklearn as skl
import joblib
import os.path

#Constants
csv_file = 'credit-data.csv'

def import_data(csv_name=csv_file):
    '''
    Loads data from a CSV file into a pandas datafram, precesses data and then
    splits that data into test and train data

    Inputs:
        csv_name: the pathway to a csv file that we will download data from
            It is set to the name of the csv_file that I will use for this
            assignment
    Outputs:
        train_df: a pandas dataframe that has the data we will train on
        test_df: a pandas dataframe that has the data we will test on
    '''

    if os.path.exists(csv_name):
        df_all_data = pd.read_csv(csv_name)
    else:
        print("Pathway to the CSV does not exist")
        return None
    df_all_data = process_data(df_all_data)
    train_df, test_df = skl.model_selection.train_test_split(df_all_data,
    	train_size=0.9, test_size=0.1)
    return train_df, test_df

def process_data(df_dirty):
	'''
    Cleans and processes data

    Inputs:
        df_dirty: a pandas dataframe

    Returns:
        df_clean: a pandas dataframe
	'''
	df_clean = df_dirty
	return df_clean

'''
Notes from sklearn

we must learn from the training set using the fit() method: can only use it once
or what we do will be overwritten 

then we can predict using the predict() method

you can save a model using joblib (maybe??)
'''
