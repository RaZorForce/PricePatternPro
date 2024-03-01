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

    few_pairs = {}
    count = 0
    for key, value in histdata.items():
        if count < 2:
            few_pairs[key] = value
            count += 1
        else:
            break

    for key , value in few_pairs.items():
        print("-"*60)
        print(f"ticker = {key}")
        print("-"*15)

        #identify all the low and highs swing points on all charts
        minima, maxima = get_min_max(value)

        pattern_data = context.pattern_detection(minima, maxima, value)

        # if pattern is not confirmed then skip ticker, start next loop
        if len(pattern_data) == 0:
            continue

        #Prepare backtesting
        strategy_data = pre_backtester(value, pattern_data )

        #Start backtesting
        trade_summary = backtester(strategy_data, key)

        #Post backtesting
        #post_backtester(strategy_data, trade_summary)

    x = 0
def pre_backtester(data: pd.DataFrame, pattern_data: pd.DataFrame) -> pd.DataFrame:

    pattern_data['confirmation_date'] = pd.to_datetime(pattern_data['confirmation_date'], format='%Y-%m-%d')

    #if confirmation date is duplicated for 2 patterns remove it
    pattern_data = pattern_data.drop_duplicates(subset=['confirmation_date'], keep='first')

    pattern_data = pattern_data.reset_index(drop = True)

    #start merging
    strategy_data = pd.merge(data, pattern_data, how ='left',
                                left_on = 'Date', right_on ='confirmation_date')

    #index was lost in merging, retrieve it
    strategy_data.index  = data.index

    # replace na with 0 in signal column
    strategy_data.signal.fillna(0, inplace=True)

    return strategy_data

def backtester(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # ------------------------------- Initial Settings -------------------------------------
    # Create dataframes for round trips, storing trades, and mtm
    trade_summary = pd.DataFrame()
    trade = pd.DataFrame()

    #Initilise current position, number of trades, cumulative pnl, stop-loss to0 and take ptofit to 100000
    current_position = 0
    trade_num = 0
    cum_pnl = 0
    stop_loss = 0
    take_profit = 100000

    # Set exit and entry flag to False
    exit_flag = False
    entry_flag = False

    # ------------------------------- Backtest Over Historical Data -------------------------------------


    for i in data.index:
        # ------------------- Positions Check ----------------------------
        # No positions
        if (current_position == 0) & (data.loc[i,'signal'] != 0):
            current_position = data.loc[i,'signal']
            entry_flag = True

        # Short positions
        elif current_position == -1:
            if (data.loc[i,'signal'] != 0):
                data.loc[i,'signal'] = 0

            if data.loc[i,'Close'] > stop_loss:
                exit_type = 'SL'
                exit_flag = True

            elif data.loc[i,'Close'] < take_profit:
                exit_type = 'TP'
                exit_flag = True

            elif data.index[-2] == i:
                if data.loc[i,'Close'] < trade.entry_price[0]:
                    exit_type = 'LP'
                elif data.loc[i,'Close'] > trade.entry_price[0]:
                    exit_type = 'LL'
                exit_flag = True

        # Long positions
        elif current_position == 1:
            if (data.loc[i,'signal'] != 0):
                data.loc[i,'signal'] = 0

            if data.loc[i,'Close'] < stop_loss:
                exit_type = 'SL'
                exit_flag = True

            elif data.loc[i,'Close'] > take_profit:
                exit_type = 'TP'
                exit_flag = True

            elif data.index[-2] == i:
                if data.loc[i,'Close'] > trade.entry_price[0]:
                    exit_type = 'LP'
                elif data.loc[i,'Close'] < trade.entry_price[0]:
                    exit_type = 'LL'
                exit_flag = True

        # ------------------------------- Entry Position Update -------------------------------------
        if entry_flag:

            #Populate the trades dataframe
            trade = pd.DataFrame(index = [0])
            trade['Symbol'] = ticker
            trade['entry_date'] = i
            trade['entry_price'] = round(data.loc[i,'Close'],2)
            trade['position'] = current_position

            stop_loss = data.loc[i,'stoploss']

            take_profit = data.loc[i,'target']

            trade_num += 1

            # Print trade details
            print(f"\033[36mTrade No: {trade_num}\033[0m | Entry Date: {i} | Entry Price : {trade.entry_price[0]}] | Position: {current_position}  | Symbol: {ticker}")

            # Set entry flag to False
            entry_flag = False

            continue

        # ------------------------------- Exit Position Update --------------------------------------
        if exit_flag:

            #Populate the trades dataframe
            trade['exit_date'] = i
            trade['exit_type'] = exit_type
            trade['exit_price'] = round(data.loc[i,'Close'],2)

            # Calculate pnl for the trade
            trade_pnl = current_position *  round( trade.exit_price[0] - trade.entry_price[0],2)

            # Calculate cumulative pnl
            cum_pnl += trade_pnl
            cum_pnl = round(cum_pnl,2)
            trade['PnL'] = trade_pnl


            trade['stoploss'] = stop_loss
            trade['target'] = take_profit

            # Add the trade logs to round trip details
            trade_summary = pd.concat([trade_summary, trade])

            # Print trade details
            print(f"Exit Type: {exit_type} | Exit Date: {i} | Exit Price: {trade.exit_price[0]}   | PnL: {trade_pnl} | Cum PnL: {cum_pnl}")
            print("-"*30)

            # Update current position to 0
            current_position = 0

            # Set the exit flag to False
            exit_flag = False

            continue

    trade_summary.reset_index(drop=True, inplace=True)

    return trade_summary


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