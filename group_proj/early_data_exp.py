'''
Esther Edith Spurlock

Early Data Exploration
'''

import pandas as pd

#Constants
rcra_csvs = ['RCRA_ENFORCEMENTS.csv',
'RCRA_EVALUATIONS.csv',
'RCRA_FACILITIES.csv',
'RCRA_NAICS.csv',
'RCRA_VIOLATIONS.csv',
'RCRA_VIOSNC_HISTORY.csv']

def download_data(csv_lst=rcra_csvs):
    '''
    Dowloads data from multiple csvs using a list of csv names
    '''
    for curr_csv in rcra_csvs:
        print(curr_csv)
        df_data = pd.read_csv(curr_csv)
        print(df_data.shape)
        print(df_data.columns)
        for col in  df_data.columns:
            curr_series = df_data[col]
            print(curr_series.describe())
