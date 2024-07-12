from datasets import DatasetDict, Dataset

def split_data_into_train_valid(data: Dataset, test_size=0.2) -> DatasetDict:
    # Divide data into Train-Valid
    divided_data = data.train_test_split(test_size=test_size, seed=42)
    train_data = divided_data["train"]
    valid_data = divided_data["test"]

    return train_data, valid_data

def tokenize_data(data: Dataset, tokenizer) -> Dataset:
    tokenizer.pad_token = tokenizer.eos_token
    max_input_length = 48
    max_target_length = 128
    
    def preprocess_function(examples):
        examples["text"] = [f"### Instruction: Create a Python script for this problem: {text} ### Response:" for text in examples["text"]]
        return examples

    data = data.map(preprocess_function)

    tokenized_inputs = tokenizer(
        data["text"],
        padding="max_length",
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt"
    )
    
    tokenized_targets = tokenizer(
        data["code"],
        padding="max_length",
        max_length=max_target_length,
        truncation=True,
        return_tensors="pt"
    )

    labels = tokenized_targets.input_ids
    labels[labels == 0] = -100

    data = Dataset.from_dict({
        "input_ids": tokenized_inputs.input_ids,
        "labels": labels
    })

    print(f"Complete tokenizing dataset with max_input_length={max_input_length}, max_target_length={max_target_length}!")
    
    return data