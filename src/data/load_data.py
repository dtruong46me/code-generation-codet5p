
from datasets import load_dataset, Dataset, concatenate_datasets
    

def ingest_data(datapath:str="mbpp") -> Dataset:
    if "," not in datapath:
        if datapath == "mbpp":
            return load_mbpp()
        elif datapath == "codealpaca":
            return load_codealpaca()
        elif datapath == "conala":
            return load_conala()
    if "," in datapath:
        datapaths = datapath.split(",")
        print(datapaths)
        all_data = Dataset.from_dict({
            "text": [],
            "code": []
        })
        for datapath in datapaths:
            if datapath == "mbpp":
                all_data = concatenate_datasets([all_data, load_mbpp()])
            elif datapath == "codealpaca":
                all_data = concatenate_datasets([all_data, load_codealpaca()])
            elif datapath == "conala":
                all_data = concatenate_datasets([all_data, load_conala()])
        return all_data

def load_mbpp() -> Dataset:
    data = load_dataset("mbpp", trust_remote_code=True)
    data = concatenate_datasets([
        data["train"],
        data["test"],
        data["validation"],
        data["prompt"]
    ])

    data = Dataset.from_dict({
        "text": data["text"],
        "code": data["code"]
    })
    return data


def load_codealpaca() -> Dataset:
    data = load_dataset("Abzu/CodeAlpacaPython", split="train")
    d = {
        "text": [],
        "code": []
    }
    for sample in data:
        prompt = sample["prompt"]
        response = sample["response"]
        if "return" in response:
            d["text"].append(prompt)
            response.replace("    ", "\t")
            d["code"].append(response)

    data = Dataset.from_dict(d)
    return data


def load_conala() -> Dataset:
    data1 = load_dataset("neulab/conala", split="train")
    d = {
        "text": [],
        "code": []
    }
    for sample in data1:
        prompt = sample["intent"]
        response = sample["snippet"]
        if "return" in response and "def" in response:
            d["text"].append(prompt)
            # response.replace("    ", "\t")
            d["code"].append(response)

    data2 = load_dataset("neulab/conala", "mined", split="train")
    for sample in data2:
        prompt = sample["intent"]
        response = sample["snippet"]
        if "return" in response and "def" in response:
            d["text"].append(prompt)
            # response.replace("    ", "\t")
            d["code"].append(response)

    data = Dataset.from_dict(d)
    
    return data

data = ingest_data("conala")
data