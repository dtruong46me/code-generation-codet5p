import logging
from abc import ABC, abstractclassmethod

class DataStrategy(ABC):
    """
    Abstract class for handling data
    """
    @abstractclassmethod
    def handle_data(self, data):
        pass