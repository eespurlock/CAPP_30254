'''
Esther Edith Spurlock

Assignment 1: Diagnostic Code

Due: 4/9/2019
'''
#imports
import pandas as pd
import sodapy
import requests

def analyze_chicago_crimes():
    '''
    This function reads in the crimes data from a CSV and then analyzes the data
    
    The following CSV comes from the the below URL
    The filters for this data are dates after 12/31/2016 11:59:59 PM and before
    12/21/2018 11:59:59 PM
    https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/
    ijzp-q8t2/data
    '''
    #Problem 1
    #download the data
    #df_crimes = pd.read_csv("Crimes_2017_2018_data.csv")
    limit = 300000
    crimes_2017 = ("https://data.cityofchicago.org/resource/6zsd-86xi.json?"
        "year=2017$limit=300000")
    client_2017 = Socrata(crimes_2017)
    dataset_id_2017 = "d62x-nvdr"
    results_2017 = client_2017.get(dataset_id_2017, limit=limit)
    crimes_2018 = ("https://data.cityofchicago.org/resource/6zsd-86xi.json?"
        "year=2018$limit=300000")
    client_2018 = Socrata(crimes_2018)
    dataset_id_2018 = "3i3m-jwuy"
    results_2018 = client_2018.get(dataset_id_2018, limit=limit)

    crimes_df_2017 = pd.DataFrame.from_records(results_2017)
    crimes_df_2018 = pd.DataFrame.from_records(results_2018)

    #This will give you counts of all the different kinds of crimes and
    #how they change
    counted_df_2017 = crimes_df_2017["Primary Type"].value_counts()
    counted_df_2018 = crimes_df_2018["Primary Type"].value_counts()
    counted_df_2017.plot()
    counted_df_2018.plot()
    #This will show you how crimes differ by block
    block_crimes_2017 = crimes_df_2017.groupby(["Block", "Primary Type"]).size()
    block_crimes_2018 = crimes_df_2018.groupby(["Block", "Primary Type"]).size()
    block_crimes_2017.plot()
    block_crimes_2018.plot()

    #Problem 2
    #This links to information about the total population, population of men
    #(for gender separation), and the allocation of military service for
    #civilian veterans
    acs_5year_data = ("https://api.census.gov/data/2017/acs/acs5?"
    	"get=B01001_001E,B01001_002E,B99212_003E,"
    	"NAME&for=block%20group:*&in=state:17%20county:031")
    client_acs = Socrata(acs_5year_data)
    results_acs = client_acs.get(limit=limit)
    acs_df = pd.DataFrame.from_records(results_acs)

    #Problem 3
    crimes_df_2017["Date"]= pd.to_datetime(crimes_df_2017["Date"])
    crimes_df_2017["Month"] = crimes_df_2017["Date"].dt.month
    crimes_df_2018["Date"]= pd.to_datetime(crimes_df_2018["Date"])
    crimes_df_2018["Month"] = crimes_df_2018["Date"].dt.month

    filt_2017 = crimes_df_2017["Month"].isin([7])
    filt_2018 = crimes_df_2018["Month"].isin([7])
    
    df_filt_2017 = crimes_df_2017[filt]
    df_filt_2018 = crimes_df_2018[filt]

    filt_2017 = df_filt_2017["Primary Type"].isin(["ROBBERY", "BATTERY", 
    	"BURGLARY", "MOTOR VEHICLE THEFT"])
    df_filt_2017 = df_filt_2017[filt]
    df_filt_2017.groupby(["Year", "Primary Type"]).size()

    filt_2018 = df_filt_2018["Primary Type"].isin(["ROBBERY", "BATTERY", 
    	"BURGLARY", "MOTOR VEHICLE THEFT"])
    df_filt_2018 = df_filt_2018[filt]
    df_filt_2018.groupby(["Year", "Primary Type"]).size()
    
    #Problem 3 - 2
    #A
    filt_2017 = crimes_df_2017["Block"].isin(["021XX S MICHIGAN AVE"])
    df_filt_2017 = crimes_df_2017[filt_2017]
    df_filt_2017.groupby(["Primary Type"]).size()

    filt_2018 = crimes_df_2018["Block"].isin(["021XX S MICHIGAN AVE"])
    df_filt_2018 = crimes_df_2018[filt_2018]
    df_filt_2018.groupby(["Primary Type"]).size()

