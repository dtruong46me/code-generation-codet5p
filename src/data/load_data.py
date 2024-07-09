from datasets import load_dataset, Dataset, concatenate_datasets


# Ingest Dataset
def ingest_data(datapath:str, split="train") -> Dataset:
    if "," not in datapath:
        if datapath=="mbpp":
            return load_mbpp(split=split)
        if datapath=="conala":
            return load_conala(split=split)
        if datapath=="codealpaca":
            return load_codealpaca(split=split)
    
    if "," not in datapath:
        all_datapaths = datapath.split(",")
        all_datasets = Dataset.from_dict({
            "text": [],
            "code": []
        })
        for datapath in all_datapaths:
            if datapath=="mbpp":
                all_datasets = concatenate_datasets([all_datasets, load_mbpp(split=split)])
            if datapath=="conala":
                all_datasets = concatenate_datasets([all_datasets, load_conala(split=split)])
            if datapath=="codealpaca":
                all_datasets = concatenate_datasets([all_datasets, load_codealpaca(split=split)])
        
        all_datasets.shuffle(seed=42)
        print(all_datasets)

        return all_datasets

# Load MBPP dataset
def load_mbpp(split="train") -> Dataset:
    if split=="valid":
        return load_dataset("google-research-datasets/mbpp", split="validation", trust_remote_code=True)
    
    data = load_dataset("google-research-datasets/mbpp", trust_remote_code=True, split=split)
    return data # -> Dataset({"text":... "code":...})

# Load CodeAlpaca dataset
def load_codealpaca(split="train") -> Dataset:
    data = load_dataset("Abzu/CodeAlpacaPython", split=split, trust_remote_code=True)

    data = data.rename_column("prompt", "text")
    data = data.rename_column("response", "code")

    data = data.filter(filter_func)
    return data

# Load Conala dataset
def load_conala(split="train") -> Dataset:
    data1 = load_dataset("conala", split=split, trust_remote_code=True)
    data2 = load_dataset("neulab/conala", "mined", split=split, trust_remote_code=True)

    data = concatenate_datasets([data1, data2])

    data = data.rename_column("intent", "text")
    data = data.rename_column("snippet", "code")

    data = data.filter(filter_func)
    return data

def filter_func(sample):
    return "return" and "def" in sample["code"]