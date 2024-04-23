import logging
from datasets import load_dataset, Dataset, concatenate_datasets

logger = logging.getLogger(__name__)

class IngestDataset:
    def __init__(self, from_huggingface:str = "mbpp") -> None:
        self.from_huggingface = from_huggingface

    def get_data(self) -> Dataset:
        logger.info(f"Loading data from {self.from_huggingface}")
        
        data = load_dataset(self.from_huggingface, trust_remote_code=True)
        data = concatenate_datasets([
            data["train"],
            data["test"],
            data["validation"],
            data["prompt"]
        ])

        logger.info("Complete loading dataset from {self.from_huggingface}")
        return data
    
def ingest_data(from_huggingface: str) -> Dataset:
    try:
        ingest_data = IngestDataset(from_huggingface)
        dataset = ingest_data.get_data()

        return dataset
    
    except Exception as e:
        logger.error(f"Error while ingesting data: {e}")
        raise e
    
# if __name__=="__main__":
#     data_path = "mbpp"
#     dataset = ingest_data(data_path)
#     print(dataset)