import logging
from datasets import load_dataset
from data_strategy import *
from load_data import *

class DataCleaning:
    def __init__(self, data, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self):
        try:
            return self.strategy.handle_data(self.data)
        
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
def clean_data(data):
    pass
        
if __name__ == '__main__':
    data = IngestDataset("code_search_net")
    strategy = DataPreprocessingStrategy()
    data_cleaning = DataCleaning(data, strategy)
    data_cleaning.handle_data()