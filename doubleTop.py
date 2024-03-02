import pandas as pd
import numpy as np
from Strategy import Strategy

class doubleTop(Strategy):

    def __init__(self):
        self.name = "Double Top"
        self.datapoints = 3
        self.bias = "Short"

    def pattern_scanner(self, minima: pd.Series, maxima: pd.Series, frequency: str ='daily') -> list:
        # To store pattern instances
        patterns = []

        #concatinate both dataframes then sort them by index
        min_max = pd.concat([minima, maxima]).sort_index()

        # Loop to iterate along the price data
        for i in range(self.datapoints, len(min_max)):

            # Store 3 local minima and local maxima points at a time in the variable 'window'
            window = min_max.iloc[i-self.datapoints:i]

            # Determine window length based on the frequency of data
            if frequency == 'daily':
                window_size = (window.index[-1] - window.index[0]).days
            else:
                window_size = (window.index[-1] - window.index[0]).seconds/60/60

            # Ensure that pattern is formed within 100 bars
            if window_size > 100:
                continue

            # Store the 3 unique points to check for conditions
            A, B, C = [window.iloc[i] for i in range(0, len(window))]

            # cond_1: To check b is in minima
            cond_1 = B in minima.values

            # cond_2: To check a,c are in maxima_prices
            cond_2 = all(x in maxima.values for x in [A, C])

            # cond_3: To check if the tops are above the neckline
            cond_3 = (B < A) and (B < C)

            # cond_4: To check if A and C are at a distance less than 1.5% away from their mean
            cond_4 = abs(A-C) <= np.mean([A, C]) * 0.1

            # Checking if all conditions are true
            if cond_1 and cond_2 and cond_3 and cond_4:

                # Append the pattern to list if all conditions are met
                patterns.append(
                    ([window.index[i] for i in range(0, len(window))]))

        print(f"Double Top pattern detected {len(patterns)} times ")

        return patterns

    def get_PriceData(self, data: pd.DataFrame, pattern_list: list ) -> pd.DataFrame :
        pattern_data = pd.DataFrame(pattern_list, columns = ['top1_date', 'neck1_date', 'top2_date'])

        # Populate the dataframe with relevant values
        pattern_data['top1_price'] = data.loc[pattern_data.top1_date, 'High'].values
        pattern_data['neck1_price'] = data.loc[pattern_data.neck1_date, 'Low'].values
        pattern_data['top2_price'] = data.loc[pattern_data.top2_date, 'High'].values

        return pattern_data

    def get_ConfDate(self, data: pd.DataFrame , pattern_data: pd.DataFrame):
        # If not empty
        if len(pattern_data) != 0:

            for x in range(0, len(pattern_data)):
                # store the data after second top in 'data_after_top2'
                data_after_top2 = data.loc[pattern_data.at[x, 'top2_date'] : ]['Close']

                try:
                    # return the short entry date if price went below the neckline
                     pattern_data.at[x,'confirmation_date'] = data_after_top2[data_after_top2 < pattern_data.at[x,'neck1_price']].index[0]

                     #pattern_data[['confirmation_date']] = pattern_data[['confirmation_date']].apply(pd.to_datetime, format='%Y-%m-%d')

                     # Store the number of days taken to generate a short entry date in the column 'time_for_confirmation'
                     pattern_data.at[x,'time_for_confirmation'] = (pattern_data.at[x,'confirmation_date'] - pattern_data.at[x,'top2_date']).days

                except:
                    # return nan if price never went below the neckline
                    pattern_data.at[x,'confirmation_date'] = np.nan

            pattern_data['signal'] = -1

            # Drop NaN values from 'hs_patterns_data'
            pattern_data.dropna(inplace=True)

            pattern_data.reset_index(drop=True, inplace = True)

            # Selecting the patterns that represent head and shoulders patterns that can be traded
            #pattern_data = pattern_data[(pattern_data['time_for_confirmation'] > 5) & ( pattern_data['time_for_confirmation'] < 30)]

        print(f"Double Top pattern confirmed {len(pattern_data)} times")

    def risk_Manager(self, pattern_data: pd.DataFrame):
        # If not empty
        if len(pattern_data) != 0:

            for x in range(0, len(pattern_data)):

                # Set stop-loss 1% above the right shoulder
                pattern_data.at[x,'stoploss'] = round(pattern_data.at[x,'top2_price']*1.01 , 2)

                # Calculate the distance between the head and the neckline
                pattern_data.at[x,'top_length'] = round(pattern_data.at[x,'top2_price'] - pattern_data.at[x,'neck1_price'] ,2)

                # Set target at a distance of head_length below the neckline
                pattern_data.at[x,'target'] = round(pattern_data.at[x,'neck1_price'] -  1 * pattern_data.at[x,'top_length'],2)