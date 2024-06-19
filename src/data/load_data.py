
from datasets import load_dataset, Dataset, concatenate_datasets

class IngestDataset:
    def __init__(self, datapath:str = "mbpp") -> None:
        self.datapath = datapath

    def get_data(self) -> Dataset:
        
        data = load_dataset(self.datapath, trust_remote_code=True)
        data = concatenate_datasets([
            data["train"],
            data["test"],
            data["validation"],
            data["prompt"]
        ])

        return data
    
def ingest_data(datapath: str) -> Dataset:
    try:
        ingest_data = IngestDataset(datapath)
        dataset = ingest_data.get_data()

        return dataset
    
    except Exception as e:
        raise e