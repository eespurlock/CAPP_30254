'''
Esther Edith Spurlock (12196692)

CAPP 30254

Assignment 3: Update the Pipeline

PY file #1: putting the pipeline together
'''

#Imports
import prep_data

#Constant for this assignment
csv_file = 'projects_2012_2013.csv'

def pipeline(csv_name=csv_file):
    '''
    Goes from the beginning to the end of the machine learning pipeline

    Inputs:
        csv_name: the pathway to a CSV file that has the data we want
            (this is initialized to the CSV file we were given for this
            assignment)

    Outputs:
        model_dict: a dictionary of the different models we have tested, the
            different parameters we have tried on them and the evaluation
            metrics we have used
    '''

    df_all_data = prep_data.import_data(csv_name)
    if df_all_data is None:
        return None
    else:
        Print('Imported')
