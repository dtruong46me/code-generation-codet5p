
from datasets import load_dataset, Dataset, concatenate_datasets
    

def ingest_data(datapath:str="mbpp") -> Dataset:
    data = load_dataset(datapath, trust_remote_code=True)
    data = concatenate_datasets([
        data["train"],
        data["test"],
        data["validation"],
        data["prompt"]
    ])

    return data

