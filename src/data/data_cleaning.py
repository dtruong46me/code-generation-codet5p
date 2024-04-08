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
    try:
        preproc_strtg = DataPreprocessingStrategy()
        data_cleanng_ = DataCleaning(data, preproc_strtg)
        processed_data = data_cleanng_.handle_data()

        split_strtg = DataDivideStrategy()
        data_cleaning_ = DataCleaning(data, split_strtg)
        train_ds, test_ds, valid_ds = data_cleaning_.handle_data()
        logging.info("Data cleaning completed!")

    except Exception as e:
        logging.error("Error while handling data")
        raise e
        
if __name__ == '__main__':
    data = IngestDataset("mbpp")
    strategy = DataPreprocessingStrategy()
    data_cleaning = DataCleaning(data, strategy)
    data_cleaning.handle_data()