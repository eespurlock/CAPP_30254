'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 5: Update the Pipeline

PY file #2: prepares the data for testing and training
'''

#Imports
import pandas as pd
import os.path
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import re

#Defined constants for the column names
PROJ_ID = 'projectid'
TEACH_ID = 'teacher_acctid'
SCHOOL_ID1 = 'schoolid'
SCHOOL_ID2 = 'school_ncesid'
LAT = 'school_latitude'
LONG = 'school_longitude'
CITY = 'school_city'
STATE = 'school_state'
METRO = 'school_metro'
DISTRICT = 'school_district'
COUNTY = 'school_county'
CHARTER = 'school_charter'
MAGNET = 'school_magnet'
PREFIX = 'teacher_prefix'
SUBJECT = 'primary_focus_subject'
AREA = 'primary_focus_area'
SUBJECT_2 = 'secondary_focus_subject'
AREA_2 = 'secondary_focus_area'
RESOURCE = 'resource_type'
POVERTY = 'poverty_level'
GRADE = 'grade_level'
PRICE = 'total_price_including_optional_support'
STUDENTS = 'students_reached'
DOUBLE = 'eligible_double_your_impact_match'
POSTED = 'date_posted'
FUNDED = 'datefullyfunded'
VAR = 'funded_in_i_days'

def import_data(csv_name):
    '''
    Imports data from a CSV file

    Inputs:
        csv_name: the pathway to a CSV file that has the data we want

    Outputs:
        df_all_data: a pandas dataframe with all of the data unchanged
    '''
    if os.path.exists(csv_name):
        df_all_data = pd.read_csv(csv_name, parse_dates=[POSTED, FUNDED],
            infer_datetime_format=True)
    else:
        print("Pathway to the CSV does not exist")
        return None
    return df_all_data

def explore_data(df_all_data, all_cols):
    '''
    Explores the data in the CSV

    Inputs:
        df_all_data: pandas dataframe with our data
        all_cols: column names in our dataframe

    Outputs:
        description_dict: a dictionary describing the data in each column
    '''
    #I'm really not sure what I'm supposed to do with this
    description_dict = {}
    for col in all_cols:
        curr_series = df_all_data[col]
        #Describes the data in the column
        description_dict[col] = curr_series.describe()
    return description_dict

def clean_data(df_all_data, all_cols):
    '''
    Cleans the data

    Inputs:
        df_all_data: pandas dataframe with our data
        all_cols: column names in our dataframe

    Outputs:
        df_all_data: a cleaned pandas dataframe
    '''
    for col in all_cols:
        ser = df_all_data[col]
        if ser.dtype in ['float64', 'int64']:
            new_col = col + '_imputed_mean'
            col_mean = ser.mean()
            df_all_data[col] = ser.fillna(col_mean)
            #If an entry is equal to the column mean we say it is imputed
            df_all_data[new_col] = df_all_data[col].apply(lambda x:\
                1 if x == col_mean else 0)
       
    return df_all_data

def generate_var_feat(df_all_data, all_cols, i, split):
    '''
    Generates the variable and features for the dataset

    Inputs:
        df_all_data: pandas dataframe with our data
        all_cols: column names in our dataframe
        i: the number of days we want something to be funded in
        split: name of the column we will use to cordon off the training and 
            testing data

    Outputs:
        df_all_data: a pandas dataframe with only the columns we will need
        variable: name of the variable column
        features: a list of the feature columns
        
    '''
    variable = VAR    
    #First, we create the variable: 0 if finded within i days and 1 if not
    df_all_data[VAR] = df_all_data[FUNDED] - df_all_data[POSTED]
    df_all_data[VAR] = df_all_data[VAR]\
        .apply(lambda x: 0 if x.days <= i else 1)
    
    #Now we need to find the features    
    all_cols = df_all_data.columns
    features = []

    for col in all_cols:
        #I do not want to include values that are different for every entry
        #I also do not want to include the variable or the dates in my features 
        if col not in [PROJ_ID, POSTED, FUNDED, VAR]:
            ser = df_all_data[col]
            if ser.dtype not in ['float64', 'int64']:
                #Find all unique values in the column
                val_unique = ser.unique()
                for val in val_unique:
                    new_col = col + "_" + str(val)
                    #Create a dummy variable on the column value
                    df_all_data[new_col] = df_all_data[col]\
                        .apply(lambda x: 1 if x == val else 0)
                    features.append(new_col)
            else:
                features.append(col)

    used_cols = features + [variable, split]
    df_all_data = drop_extra_columns(df_all_data, used_cols, all_cols)

    return df_all_data, variable, features

def drop_extra_columns(df_all_data, col_list, all_cols):
    '''
    Drops columns from the dataframe we are not going to use in analysis

    I am looking ahead with this function. I do not expect to use it for this
    assignment.

    Inputs:
        df_all_data: a pandas dataframe
        col_list: list of columns we are going to use
        all_cols: list of all columns in the dataframe

    Outputs:
        df_all_data: a pandas dataframe
    '''
    to_drop = []
    for col in all_cols:
        if col not in col_list:
            to_drop.append(col)
    if to_drop != []:
        df_all_data = df_all_data.drop(to_drop, axis=1)
    return df_all_data
