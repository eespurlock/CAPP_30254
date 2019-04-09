'''
Esther Edith Spurlock

Assignment 1: Diagnostic Code

Due: 4/9/2019
'''
#imports
import pandas as pd
import sodapy
from sodapy import Socrata
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
    #Originally, I was able to download the data using a CSV (described above)
    #Below is my attempt to download the data using an API. It is unsuccessful.
    #I am going to apologize in advance to the grader who has the misfortune of
    #Looking through this. It is a hot mess and I'm not going to try to hide it.

    crimes = ("https://data.cityofchicago.org/resource/")
    client = Socrata(crime, None)
    dataset_id_2017 = "d62x-nvdr"
    results_2017 = dwonload_all_data(client, dataset_id_2017)
    dataset_id_2018 = "3i3m-jwuy"
    results_2018 = dwonload_all_data(client, dataset_id_2018)

    crimes_df_2017 = pd.DataFrame.from_records(results_2017)
    crimes_df_2018 = pd.DataFrame.from_records(results_2018)
    
    #Puts in the month column
    crimes_df_2017["Date"]= pd.to_datetime(crimes_df_2017["Date"])
    crimes_df_2017["Month"] = crimes_df_2017["Date"].dt.month
    crimes_df_2018["Date"]= pd.to_datetime(crimes_df_2018["Date"])
    crimes_df_2018["Month"] = crimes_df_2018["Date"].dt.month

    #The following commands all have tables in my writeup
    #I do not have the graphs in my writeup because I don't know how to put
    #graphs in these files.
    #How did I make this writeup when my code doesn't work?
    #I'm glad you asked! I used the data I got in from a CSV
    #This will give you counts of all the different kinds of crimes and
    #how they change
    counted_df_2017 = crimes_df_2017["Primary Type"].value_counts()
    counted_df_2018 = crimes_df_2018["Primary Type"].value_counts()
    counted_df_2017.plot()
    counted_df_2018.plot()

    #Breaks down total crime by month
    month_crimes_2017 = crimes_df_2017.groupby(["Month"]).size()
    month_crimes_2018 = crimes_df_2018.groupby(["Month"]).size()
    month_crimes_2017.plot()
    month_crimes_2018.plot()

    #Breaks down kinds of crime by month
    month_type_crimes_2017 = crimes_df_2017.groupby(["Month", "Primary Type"])\
        .size()
    month_type_crimes_2018 = crimes_df_2018.groupby(["Month", "Primary Type"])\
        .size()

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
        "NAME&for=block%20group:*&in=state:17%20county:031"
        "&key=5f73e259b50f6b42c86fddc370dda1f276928cc1")
    client_acs = Socrata(acs_5year_data, None)
    acs_id = "acs5"
    results_acs = dwonload_all_data(client_acs, acs_id)
    acs_df = pd.DataFrame.from_records(results_acs)

    #This data is problematic because it is from 2010, but it was the most
    #recent tract data I could find
    census_tract = ("https://data.cityofchicago.org/Facilities-Geographic-"
        "Boundaries/Boundaries-Census-Tracts-2010/")
    clinet_tract = Socrata(census_tract, None)
    tract_id = "5jrd-6zik"
    results_tract = dwonload_all_data(clinet_tract, tract_id)
    tract_df = pd.Dataframe.from_records(results_tract)
    tract_df.rename(\
        index=str, columns={"TRACTCE10": "tract", "COMMAREA": "Community Area"})

    #The following gives names for all the community areas
    community_area = ("https://data.cityofchicago.org/Facilities-Geographic-"
        "Boundaries/Boundaries-Community-Areas-current-/")
    client_area = Socrata(community_area, None)
    area_id = "cauq-8yn6"
    results_area = dwonload_all_data(client_area, area_id)
    area_df = pd.DataFrame.from_records(results_area)
    area_df.rename(index=str, columns={"AREA_NUMBE": "Community Area"})

    #Merging all of our 'beautiful' data
    tract_with_area = area_df.merge(tract_df, how='left', on='Community Area')
    acs_with_geo = acs_df.merge(tract_with_area, how='left', on="tract")
    crimes_with_acs_2017 = acs_with_geo\
        .merge(crimes_df_2017, how='left', on='Community Area')
    crimes_with_acs_2018 = acs_with_geo\
        .merge(crimes_df_2018, how='left', on='Community Area')

    #Code for question 1 / 3
    filt_battery_2017 = crimes_with_acs_2017["Primary Type"].isin(["BATTERY"])
    df_filt_2017 = crimes_with_acs_2017[filt_battery_2017]
    df_filt_2017.groupby(["COMMUNITY","B01001_001E","B01001_002E",
        "B99212_003E"]).size()

    filt_battery_2018 = crimes_with_acs_2018["Primary Type"].isin(["BATTERY"])
    df_filt_2018 = crimes_with_acs_2018[filt_battery_2018]
    df_filt_2018.groupby(["COMMUNITY","B01001_001E","B01001_002E",
        "B99212_003E"]).size()

    #Code for question 2 / 3
    filt_hom_2017 = crimes_with_acs_2017["Primary Type"].isin(["HOMICIDE"])
    df_filt_2017 = crimes_with_acs_2017[filt_hom_2017]
    df_filt_2017.groupby(["COMMUNITY","B01001_001E","B01001_002E",
        "B99212_003E"]).size()

    filt_hom_2018 = crimes_with_acs_2018["Primary Type"].isin(["HOMICIDE"])
    df_filt_2018 = crimes_with_acs_2018[filt_hom_2018]
    df_filt_2018.groupby(["COMMUNITY","B01001_001E","B01001_002E",
        "B99212_003E"]).size()

    #Code for question 4
    filt_mult_2017 = crimes_with_acs_2017["Primary Type"].isin(
        ["DECEPTIVE PRACTICE", "SEX OFFENSE"])
    df_filt_2017 = crimes_with_acs_2017[filt_mult_2017]
    df_filt_2017.groupby(["Primary Type","COMMUNITY","B01001_001E",
        "B01001_002E","B99212_003E"]).size()

    filt_mult_2018 = crimes_with_acs_2018["Primary Type"].isin(
        ["DECEPTIVE PRACTICE", "SEX OFFENSE"])
    df_filt_2018 = crimes_with_acs_2018[filt_mult_2018]
    df_filt_2018.groupby(["Primary Type", "COMMUNITY","B01001_001E",
        "B01001_002E", "B99212_003E"]).size()

    #Below is the code for how I answered some of the questions in the writeup
    #Problem 3 - 1
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

def dwonload_all_data(client, dataset_id):
    '''
    Downloads a large amount of data from an API

    Inputs:
        client: the original link to the data
        dataset_id: the unique id for the dataset
    Returns:
        full_dataset: all of the data we have downloaded
    '''
    #This is the function that keeps 
    limit = 50000
    offset = 0
    full_dataset = []
    more_results = client.get(dataset_id, limit=limit, offset=offset)

    while len(more_results) > 0:
        full_dataset.extend(more_results)
        offset += limit
        more_results = client.get(dataset_id, limit=limit, offset=offset)

    return full_dataset
