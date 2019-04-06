'''
Esther Edith Spurlock

Assignment 1: Diagnostic Code

Due: 4/9/2019
'''
#imports
import pandas as pd

def read_crimes_csv():
    '''
    This function reads in the crimes data from a CSV and then analyzes the data
    
    The following CSV comes from the the below URL
    The filters for this data are dates after 12/31/2016 11:59:59 PM and before
    12/21/2018 11:59:59 PM
    https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/
    ijzp-q8t2/data
    '''
    #Problem 1
    df_crimes = pd.read_csv("Crimes_2017_2018_data.csv")
    #This will give you counts of all the different kinds of crimes
    counted_df = df_crimes["Primary Type"].value_counts()
    counted_df.plot()
    #This will show you how crimes change over time
    year_crimes = df_crimes.groupby(["Year","Primary Type"]).size()
    year_crimes.plot()
    #This will show you how crimes differ by block
    block_crimes = df_crimes.groupby(["Block", "Primary Type"]).size()
    block_crimes.plot()

    #Problem 3

    df_crimes["Date"]= pd.to_datetime(df_crimes["Date"])
    df_crimes["Month"] = df_crimes["Date"].dt.month
    filt = df_crimes["Month"].isin([7])
    df_filt = df_crimes[filt]
    filt = df_filt["Primary Type"].isin(["ROBBERY", "BATTERY", "BURGLARY",
    	"MOTOR VEHICLE THEFT"])
    df_filt = df_filt[filt]
    df_filt.groupby(["Year", "Primary Type"]).size()