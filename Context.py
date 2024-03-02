#from abc import ABC, abstractmethod
import pandas as pd
from Strategy import Strategy

class Context():

    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy
        self._name = strategy.name
        self._datapoints = strategy.datapoints
        self._bias = strategy.bias

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def pattern_detection(self, minima: pd.Series, maxima: pd.Series, dataframe: pd.DataFrame) -> pd.DataFrame:

        # Run scanner
        pattern_data = self._strategy.pattern_scanner(minima, maxima)

        #collect the pattern price points
        pattern_data = self.strategy.get_PriceData(dataframe, pattern_data)

        # Store the information for confirmation with the rest of the pattern data
        self.strategy.get_ConfDate(dataframe, pattern_data)

        #Determine the exit levels
        self.strategy.risk_Manager(pattern_data)

        return pattern_data