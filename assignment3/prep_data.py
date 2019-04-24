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


def import_data(csv_name):
    '''
    Imports data from a CSV file

    Inputs:
        csv_name: the pathway to a CSV file that has the data we want

    Outputs:
        df_all_data: a pandas dataframe with all of the data unchanged
    '''
    if os.path.exists(csv_name):
        col_types = {PROJ_ID: str, TEACH_ID: str, SCHOOL_ID1: str,
            SCHOOL_ID2: int, LAT: float, LONG: float, CITY: str, STATE: str,
            METRO: str, DISTRICT: str, COUNTY: str, CHARTER: str, MAGNET: str,
            PREFIX: str, SUBJECT: str, AREA: str, SUBJECT_2: str, AREA_2: str,
            RESOURCE: str, POVERTY: str, GRADE: str, PRICE: float,
            STUDENTS: int, DOUBLE: str, POSTED: str, FUNDED: str}
        df_all_data = pd.read_csv(csv_name, dtype=col_types)
    '''
    else:
        print("Pathway to the CSV does not exist")
        return None
    '''
    for datecol in [POSTED, FUNDED]:
        df_all_data[datecol] = pd.to_datetime(df_all_data[datecol])
    return df_all_data
