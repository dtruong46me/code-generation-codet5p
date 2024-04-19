import logging
from abc import ABC, abstractclassmethod
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class DataStrategy(ABC):
    """
    Abstract class for handling data
    """
    @abstractclassmethod
    def handle_data(self, data, *args) -> None:
        pass


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: Dataset, *args) -> DatasetDict:
        try:
            # Divide data into Train-Test
            divided_data = data.train_test_split(test_size=0.3)
            train_data = divided_data["train"]
            test_data = divided_data["test"]

            # Divide data into Test-Valid
            test_valid_data = test_data.train_test_split(test_size=0.5)
            test_data = test_valid_data["train"]
            valid_data = test_valid_data["test"]
            
            logger.info("Complete spliting dataset!")
            return DatasetDict({
                "train": train_data,
                "test": test_data,
                "valid": valid_data
            })

        except Exception as e:
            logger.error(f"Error in dividing data: {e}")
            raise e

class DataTokenizingStrategy(DataStrategy):
    def handle_data(self, data: Dataset, tokenizer) -> Dataset:
        try:
            tokenizer.pad_token = tokenizer.eos_token
            max_input_length = 600
            max_target_length = 320


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
            
            data = Dataset.from_dict({
                "input_ids": tokenized_inputs.input_ids,
                "attention_mask": tokenized_inputs.attention_mask,
                "labels": tokenized_targets.input_ids
            })

            logger.info("Complete tokenizing dataset")
            
            return data

        except Exception as e:
            logger.error("Error while preprocessing data")
            raise e
        
# if __name__=='__main__':
#     data_path = "mbpp"
#     dataset = ingest_data(data_path)
#     print(dataset)

#     checkpoint = "Salesforce/codet5-base"
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     tokenizer_strategy = DataTokenizingStrategy()
#     dataset = tokenizer_strategy.handle_data(dataset, tokenizer)
#     print(dataset)
#     print(dataset[0])

#     print()
#     print(tokenizer.decode(dataset[0]["input_ids"]))
#     print(tokenizer.decode(dataset[0]["attention_mask"]))
#     print(tokenizer.decode(dataset[0]["labels"]))