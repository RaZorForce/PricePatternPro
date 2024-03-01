from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):

    @abstractmethod
    def pattern_scanner(self, minima: pd.Series, maxima: pd.Series, frequency: str ='daily') -> list:
        pass

    @abstractmethod
    def get_PriceData(self, data: pd.DataFrame, pattern_list: list) -> pd.DataFrame :
        pass

    @abstractmethod
    def get_ConfDate(self, data: pd.DataFrame , pattern_data: pd.DataFrame):
        pass

    @abstractmethod
    def risk_Manager(self, data: pd.DataFrame):
        pass