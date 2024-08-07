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
    
    if "," in datapath:
        all_datapaths = datapath.split(",")
        print("[+] Loading dataset from:", all_datapaths)
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

        return all_datasets

# Load MBPP dataset
def load_mbpp(split="train") -> Dataset:
    data = load_dataset("google-research-datasets/mbpp", trust_remote_code=True, split=split)
    data = data.remove_columns(['task_id', 'test_list', 'test_setup_code', 'challenge_test_list'])
    data = data.map(lambda example: {"code": example["code"].replace("    ", "\t")})
    print(".............\ngoogle-research-datasets/mbpp\n", data)
    print("Sample:\n")
    print("[+] Text:", data[0]["text"], end="\n\n")
    print("[+] Code:\n", data[0]["code"])
    return data # -> Dataset({"text":... "code":...})

# Load CodeAlpaca dataset
def load_codealpaca(split="train") -> Dataset:
    data = load_dataset("Abzu/CodeAlpacaPython", split=split, trust_remote_code=True)

    data = data.rename_column("prompt", "text")
    data = data.rename_column("response", "code")
    data = data.map(lambda example: {"code": example["code"].replace("    ", "\t")})
    data = data.filter(filter_func)
    print(".............\nAbzu/CodeAlpacaPython\n", data)
    print("Sample:\n")
    print("[+] Text:", data[0]["text"], end="\n\n")
    print("[+] Code:\n", data[0]["code"])
    return data

# Load Conala dataset
def load_conala(split="train") -> Dataset:
    data1 = load_dataset("neulab/conala", split=split, trust_remote_code=True)
    data1 = data1.remove_columns(['question_id', 'rewritten_intent'])
    data1 = data1.select([x for x in range(0, 500, 2)])
    
    data2 = load_dataset("neulab/conala", "mined", split=split, trust_remote_code=True)
    data2 = data2.remove_columns(['question_id', 'parent_answer_post_id', 'prob', 'id'])
    data2 = data2.select([x for x in range(0, 500, 2)])
    
    data = concatenate_datasets([data1, data2])

    data = data.rename_column("intent", "text")
    data = data.rename_column("snippet", "code")
    data = data.map(lambda example: {"code": example["code"].replace("    ", "\t")})

    print(".............\nneulab/conala\n", data)
    print("Sample:\n")
    print("[+] Text:", data[0]["text"], end="\n\n")
    print("[+] Code:\n", data[0]["code"])
    return data

def filter_func(sample):
    return "return" and "def" in sample["code"]