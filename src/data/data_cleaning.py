import logging
from data_strategy import *
from load_data import *

class DataCleaning:
    def __init__(self, data, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self, *args):
        try:
            return self.strategy.handle_data(self.data, *args)
        
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
def clean_data(data, *args):
    try:
        tokenizing_strtg = DataTokenizingStrategy()
        data_cleanng_ = DataCleaning(data, tokenizing_strtg)
        tokenized_data = data_cleanng_.handle_data(*args)

        split_strtg = DataDivideStrategy()
        data_cleaning_ = DataCleaning(tokenized_data, split_strtg)
        tokenized_data = data_cleaning_.handle_data(*args)
        logging.info("Data cleaning completed!")
        return tokenized_data

    except Exception as e:
        logging.error("Error while handling data")
        raise e
        
if __name__ == '__main__':
    data = IngestDataset("mbpp")
    strategy = DataTokenizingStrategy()
    data_cleaning = DataCleaning(data, strategy)
    data_cleaning.handle_data()