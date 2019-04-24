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

#Defined constants for this assignment
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
VAR = 'days_between_posting_and_funding'

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
        !!!TBD!!!
    '''

def clean_data(df_all_data, all_cols):
    '''
    Cleans the data

    Inputs:
        df_all_data: pandas dataframe with our data
        all_cols: column names in our dataframe

    Outputs:
        df_all_data: a cleaned pandas dataframe
    '''

def generate_var_feat(df_all_data, all_cols):
    '''
    Generates the variable and features for the dataset

    Inputs:
        df_all_data: pandas dataframe with our data
        all_cols: column names in our dataframe

    Outputs:
        df_used_data: a pandas dataframe with only the columns we will need
        variable: name of the variable column
        features: a list of the feature columns
        split: name of the column we will use to cordon off the training and 
            testing data
    '''
    variable = VAR
    split = POSTED
    
    #First, we find the variable
    #This code is written with help from Stack Overflow
    #https://stackoverflow.com/questions/151199/how-to-calculate-number-of-days-
    #between-two-given-dates
    #and
    #https://stackoverflow.com/questions/33680666/creating-a-new-column-by-using
    #-lambda-function-on-two-existing-columns

    df_all_data[VAR] = df_all_data.apply(lambda x: (x[FUNDED] - x[POSTED]).days)
    features = all_cols
    used_cols = features + [variable, split]
    all_cols = df_all_data.columns

    if len(used_cols) < len(all_cols):
        df_all_data = drop_extra_columns(df_all_data, used_cols, all_cols)

    return df_used_data, variable, features, split

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
