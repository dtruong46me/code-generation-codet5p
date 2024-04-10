import logging
from datasets import load_dataset
from datasets import Dataset, concatenate_datasets

class IngestDataset:
    def __init__(self, from_huggingface:str = "mbpp") -> None:
        self.from_huggingface = from_huggingface

    def get_data(self) -> Dataset:
        logging.info(f"Loading data from {self.from_huggingface}")
        data = load_dataset(self.from_huggingface, trust_remote_code=True)
        data = concatenate_datasets([
            data["train"],
            data["test"],
            data["validation"],
            data["prompt"]
        ])
        return data
    
def ingest_data(from_huggingface: str) -> Dataset:
    try:
        ingest_data = IngestDataset(from_huggingface)
        dataset = ingest_data.get_data()

        return dataset
    
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
    
if __name__=="__main__":
    data_path = "mbpp"
    dataset = ingest_data(data_path)
    print(dataset)