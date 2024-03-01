import pandas as pd
import numpy as np
import os
import glob
from typing import Tuple
from scipy.signal import argrelextrema

from Context import Context

from doubleBottom import doubleBottom
from doubleTop import doubleTop
from headAndShoulders import headAndShoulders
from headAndShoulders_inverse import headAndShoulders_inverse

def main():
    context = Context(doubleBottom())
    # Initialize variables
    minima, maxima= {}, {}
    pattern_data = pd.DataFrame()

    strategy_data_global = {}
    trade_summary_global = pd.DataFrame()

    # Read historical data files
    histdata = read_Daily_historical_files()

# =============================================================================
#     first_two_pairs = {}
#     count = 0
#     for key, value in histdata.items():
#         if count < 2:
#             few_pairs[key] = value
#             count += 1
#         else:
#             break
# =============================================================================

    for key , value in histdata.items():
        print("-"*60)
        print(f"ticker = {key}")
        print("-"*15)

        #identify all the low and highs swing points on all charts
        minima, maxima = get_min_max(value)

        pattern_data = context.pattern_detection(minima, maxima, value)

        # if pattern is not confirmed then skip ticker, start next loop
        if len(pattern_data) == 0:
            continue

    x = 0

def read_Daily_historical_files() ->dict :
    # change directory
    os.chdir("C:\\X\\Workspaces\\ALGO\\Data\\Historical\\Daily")

    # Collect all filenames in current directory
    filenames = glob.glob("*_Daily_Bars.csv")

    histdata = { } #empty dictionary NO KEYS NO VALUES

    for filename in filenames:
        # open, read and store content in a dataframe
        with open("C:\\X\\Workspaces\\ALGO\\Data\\Historical\\Daily\\" + filename, "r") as file:
            df = pd.read_csv(file,index_col=0)
            df.index = pd.to_datetime(df.index, format='%Y%m%d')

        # round each price column
        for i in range (0,4):
            df.iloc[:,i] = round(df.iloc[:,i],2)

        # store data in a dict of dataframes
        ticker = filename.split("_")[0]
        histdata[ticker] = df

    return histdata

def get_min_max(df :pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    argrel_window = 5

    #use the argrelextrema to compute the local minima and maxima points
    local_min = argrelextrema(df.iloc[:-argrel_window]['Low'].values,
                              np.less, order=argrel_window)[0]

    local_max = argrelextrema(df.iloc[:-argrel_window]['High'].values,
                              np.greater, order=argrel_window)[0]

    #store the minima and maxima values in a dataframe
    return  df.iloc[local_min].Low,  df.iloc[local_max].High

if __name__ == "__main__":
    main()