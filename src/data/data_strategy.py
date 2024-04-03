import logging
from abc import ABC, abstractclassmethod
from datasets import load_dataset

class DataStrategy(ABC):
    """
    Abstract class for handling data
    """
    @abstractclassmethod
    def handle_data(self, data):
        pass


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data):
        return
    

class DataPreprocessingStrategy(DataStrategy):
    def handle_data(self, data):
        return