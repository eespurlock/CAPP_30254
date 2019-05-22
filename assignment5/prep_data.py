'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 3: Update the Pipeline

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
VAR = 'funded_in_60_days'

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
    #At this point in the class, I'm really not sure what I'm supposed
    #to do with this
    #!!!I need to figure out what to do here!!!
    description_dict = {}
    for col in all_cols:
        print(col)
        curr_series = df_all_data[col]
        #Describes the data in the column
        description_dict[col] = curr_series.describe()
        #Plots the data in the column and saves the plot
        #plt.hist(col, data=df_all_data)
        #plt.savefig(col)
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
            #!!!See if I have time to make this imputation better!!!
            df_all_data[col] = ser.fillna(col_mean)
            #If an entry is equal to the column mean we say it is imputed
            df_all_data[new_col] = df_all_data[col].apply(lambda x:\
                1 if x == col_mean else 0)
       
    return df_all_data

def generate_var_feat(df_all_data, all_cols):
    '''
    Generates the variable and features for the dataset

    Inputs:
        df_all_data: pandas dataframe with our data
        all_cols: column names in our dataframe

    Outputs:
        df_all_data: a pandas dataframe with only the columns we will need
        variable: name of the variable column
        features: a list of the feature columns
        split: name of the column we will use to cordon off the training and 
            testing data
    '''
    variable = VAR
    split = POSTED
    
    #First, we create the variable: 0 of finded within 60 days and 1 if not
    df_all_data[VAR] = df_all_data[FUNDED] - df_all_data[POSTED]
    df_all_data[VAR] = df_all_data[VAR]\
        .apply(lambda x: 0 if x.days <= 60 else 1)
    
    #Now we need to find the features    
    all_cols = df_all_data.columns
    features = []

    for col in all_cols:
        if col != PROJ_ID:
            ser = df_all_data[col]
            if ser.dtype not in ['float64', 'int64']:
                #Find all unique values in the column
                val_unique = ser.unique()
                for val in val_unique:
                    new_col = col + "_" + val
                    #Create a dummy variable on the column value
                    
                    df_all_data[new_col] = df_all_data[col]\
                        .apply(lambda x: 1 if x == val else 0)
                    ser = df_all_data[new_col]
                    #!!!Why isn't this working???
                    if ser.nunique() > 1:
                        features.append(new_col)
            else:
                if ser.nunique() > 1:
                    features.append(col)

    used_cols = features + [variable, split]
    df_all_data = drop_extra_columns(df_all_data, used_cols, all_cols)

    return df_all_data, variable, features, split

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
