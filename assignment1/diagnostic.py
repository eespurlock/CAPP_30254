'''
Esther Edith Spurlock

Assignment 1: Diagnostic Code

Due: 4/9/2019
'''
#imports
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import seaborn as sns

#Constants

#The following CSV comes from the the below URL
#The filters for this data are dates after 12/31/2016 11:59:59 PM and before
#12/21/2018 11:59:59 PM
#https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2/
#data
CSV_NAME = "Crimes_2017_2018_data.csv"

#The following are the constants for the column names for problem 1
ID = "ID"
CASE_NUM = "Case Number"
DATE = "Date"
BLOCK = "Block"
IUCR = "IUCR"
PRIMARY = "Primary Type"
DES = "Description"
LOC_DES = "Location Description"
ARREST = "Arrest"
DOMESTIC = "Domestic"
BEAT = "Beat"
DISTRICT = "District"
WARD = "Ward"
COMMUNITY = "Community Area"
FBI_CODE = "FBI Code"
X_CO = "X Coordinate"
Y_CO = "Y Coordinate"
YEAR = "Year"
UPDATED = "Updated On"
LAT = "Latitude"
LON = "Longitude"
LOCATION = "Location"

#Problem 1
def read_crimes_csv(csvfile=CSV_NAME):
    '''
    This function reads in the crimes data from a CSV and then cleans the data
    
    Input: csvfile: the name of a CSV file
    '''
    if os.path.exists(csvfile):
        df_crimes = pd.read_csv(csvfile)
    else:
        #Should exit out of the function if the path doesn't exist.
    	print("No path to this CSV")
    	return None

    summary_stats(df_crimes)

def summary_stats(df_crimes):
	'''
	Creates summary statistics from the crime datarame

	Input: df_crimes: a pandas dataframe
	'''
	#This will give you counts of all the different kinds of crimes
	counted_df = df_crimes[PRIMARY].value_counts()
	#This will show you how crimes change over time
	year_crimes = df_crimes.groupby(["Year","Primary Type"]).size()
	#This will show you how crimes differ by block
	block_crimes = df_crimes.groupby(["Block", "Primary Type"]).size()

	return counted_df, year_crimes, block_crimes

#Problem 2