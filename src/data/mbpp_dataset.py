
import logging
from data_strategy import *

class MBPP_Dataset:
    def __init__(self, datapath):
        self.data = ingest_data(datapath)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def split_data(self, test_size=0.2):
        return
    
    def tokenize_data(self, tokenizer):
        return