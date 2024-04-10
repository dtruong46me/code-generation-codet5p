import logging
from datasets import Dataset

import sys
import os
path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)
from data_strategy import *
from load_data import *

checkpoint = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

class DataCleaning:
    def __init__(self, data: Dataset, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self, *args):
        try:
            return self.strategy.handle_data(self.data, *args)
        
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
def clean_data(data: Dataset, *args):
    try:
        tokenizing_strtg = DataTokenizingStrategy()
        data_cleanng_ = DataCleaning(data, tokenizing_strtg)
        tokenized_data = data_cleanng_.handle_data(tokenizer)

        split_strtg = DataDivideStrategy()
        data_cleaning_ = DataCleaning(tokenized_data, split_strtg)
        tokenized_data = data_cleaning_.handle_data(*args)
        logging.info("Data cleaning completed!")
        return tokenized_data

    except Exception as e:
        logging.error("Error while handling data")
        raise e
    
if __name__=='__main__':
    checkpoint = "Salesforce/codet5-base"
    datapath = "mbpp"

    data = load_dataset(datapath)
    data = clean_data(data)