import logging
from datasets import load_dataset
from data_strategy import *

class IngestDataset:
    def __init__(self, from_huggingface="code_search_net", planguage="python"):
        self.from_huggingface = from_huggingface
        self.planguage = planguage

    def get_data(self):
        logging.info(f"Loading data from {self.from_huggingface}")
        data = load_dataset(self.from_huggingface, self.planguage, trust_remote_code=True)
        return data
    
def ingest_data(from_huggingface: str, planguage: str):
    try:
        ingest_data = IngestDataset(from_huggingface, planguage)
        dataset = ingest_data.get_data()

        return dataset
    
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
    
if __name__=="__main__":
    data_path = "code_search_net"
    program_language = "ruby"
    dataset = ingest_data(data_path, program_language)
    print(dataset)