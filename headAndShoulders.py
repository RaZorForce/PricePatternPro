import pandas as pd
import numpy as np
from Strategy import Strategy

class headAndShoulders(Strategy):

    def __init__(self):
        self.name = "Head and Shoulders"
        self.datapoints = 5
        self.bias = "Short"

    def pattern_scanner(self, minima: pd.Series, maxima: pd.Series, frequency: str ='daily') -> list:

        # To store pattern instances
        patterns = []

        #concatinate both dataframes then sort them by index
        min_max = pd.concat([minima, maxima]).sort_index()

        #loop to iterate along the price data
        for i in range(self.datapoints, len(min_max)):

            #store 5 local minima and local maxima points at a time in the variable window
            window = min_max.iloc[i-self.datapoints:i]

            #determine window length based on the frequency of data
            if frequency == 'daily':
                window_size = (window.index[-1] - window.index[0]).days
            else:
                window_size = (window.index[-1] - window.index[0]).seconds/60/60

            #Ensure that pattern is formed within 100 bars
            if window_size > 100:
                continue

            #store the 5 unique points to check for conditions
            A, B, C, D, E = [window.iloc[i] for i in range(0, len(window))]

            #cond_1: To check A,C,E are in maxima prices
            cond_1 = all(x in maxima.values for x in [A, C, E])

            #cond_2: To check B,D are in minima prices
            cond_2 = all(x in minima.values for x in [B, D])

            #cond_3: To check if the shoulders are above the neckline and the head is above the shoulders
            cond_3 = (C>A) and (C>E) and (A>B) and (A>D) and (E>B) and (E>D)

            #cond_4: To check if A and E (and also B and C) are at a distance less than 50% away from their mean
            cond_4 = (abs(A-E) <= np.mean([A,E])*0.1  and abs(B-D) <= np.mean([B,D])*0.1)

            #checking if all conditions are true
            if cond_1 and cond_2 and cond_3 and cond_4:
                patterns.append(([window.index[i] for i in range(0, len(window))]))

        print(f"Head and Shoulders pattern detected {len(patterns)} times ")

        return patterns

    def get_PriceData(self, data: pd.DataFrame, pattern_list: list) -> pd.DataFrame :
        pattern_data = pd.DataFrame(pattern_list, columns = ['sh1_date', 'neck1_date', 'head_date', 'neck2_date', 'sh2_date'])

        #populate the dataframe with relevant values
        pattern_data['sh1_price']   = data.loc[pattern_data.sh1_date,'High'].values
        pattern_data['neck1_price'] = data.loc[pattern_data.neck1_date,'Low'].values
        pattern_data['head_price']  = data.loc[pattern_data.head_date,'High'].values
        pattern_data['neck2_price'] = data.loc[pattern_data.neck2_date,'Low'].values
        pattern_data['sh2_price']   = data.loc[pattern_data.sh2_date,'High'].values

        return pattern_data

    def get_ConfDate(self, data: pd.DataFrame, pattern_data: pd.DataFrame):

        # If not empty
        if len(pattern_data) != 0:
            for x in range(0, len(pattern_data)):
                # Initialise data after the formation of shoulder2
                data_after_sh2 = data.loc[ pattern_data.at[ x,'sh2_date'] :  , ]['Close']

                try:
                    # return the long entry date if price went below the neckline
                    pattern_data.at[x, 'confirmation_date'] = data_after_sh2[data_after_sh2 < pattern_data.at[x,'neck2_price']] .index[0]

                    #pattern_data[['confirmation_date']] = pattern_data[['confirmation_date']].apply(pd.to_datetime, format='%Y-%m-%d')

                    # Store the number of days taken to generate a short entry date in the column 'time_for_confirmation'
                    pattern_data.at[x,'time_for_confirmation'] = (pattern_data.at[x,'confirmation_date'] - pattern_data.at[x,'sh2_date']).days

                except IndexError:
                    # Assign NaN if prices were not able to close below the neckline
                    pattern_data.at[x, 'confirmation_date'] = np.nan

            pattern_data['signal'] = -1

            # Drop NaN values from 'hs_patterns_data'
            pattern_data.dropna(inplace=True)

            pattern_data.reset_index(drop=True, inplace = True)

            # Selecting the patterns that represent head and shoulders patterns that can be traded
            #pattern_data = pattern_data[(pattern_data['time_for_confirmation'] > 5) & ( pattern_data['time_for_confirmation'] < 30)]

        print(f"Head and Shoulders pattern confirmed {len(pattern_data)} times")

    def risk_Manager(self, pattern_data: pd.DataFrame):
        # If not empty
        if len(pattern_data) != 0:

            for x in range(0, len(pattern_data)):

                # Set stop-loss 1% above the right shoulder
                pattern_data.at[x,'stoploss'] = round(pattern_data.at[x,'sh2_price']*0.99 , 2)

                # Calculate the distance between the head and the neckline
                pattern_data.at[x,'head_length'] = round(pattern_data.at[x,'neck2_price'] - pattern_data.at[x,'head_price'] ,2)

                # Set target at a distance of head_length below the neckline
                pattern_data.at[x,'target'] = round(pattern_data.at[x,'neck2_price'] +  1 * pattern_data.at[x,'head_length'],2)