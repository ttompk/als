'''
EDA functions
ALS project
'''
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import webbrowser, os

def feature_report(df):
    '''
    print a report of every feature
    '''
    profile = ProfileReport(df, bins=30)
    profile.to_file(outputfile="output.html")
    webbrowser.open('file://' + os.path.realpath("output.html"))
    #end function

    
def dups(df):
    '''
    finds duplicated values in dataframe, requires 'subject_id'
    '''
    bill = df.copy()
    dup_cols = []
    bill['test_dup']= np.ones(len(df))
    g = bill.groupby('subject_id').count()
    print("Length df: {}".format(len(df)))
    print("Length g: {}".format(len(g)))
    for c in range(len(g.columns)):
        mask = (g.iloc[:,c]>1)
        dup_cols.append(sum(mask))
    print("Number subjects duplicated: {}".format(np.max(dup_cols)))
    bill.drop('test_dup', axis=1, inplace=True)
    if np.max(dup_cols)>=1:
        return g[mask]
    # end function

    
def to_zero_one(df, col_list):
    '''
    convert nans to 0
    input:
        df = dataframe
        col_list = list of columns to apply the change
    '''
    for col in col_list:
        df[col].fillna(int(0), inplace=True)
    # end function
    
    
def ints(df, col_list):
    '''
    makes column integer values
    input:
        df = dataframe
        col_list = list of columns to apply the change
    '''
    for col in col_list:
        df[col] = df[col].astype(int)
    # end function
    
    


