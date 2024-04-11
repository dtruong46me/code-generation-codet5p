import logging
from abc import ABC, abstractclassmethod
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from load_data import ingest_data

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
            data = data.train_test_split(test_size=0.2)
            logger.info("Complete spliting dataset!")
            return data

        except Exception as e:
            logger.error(f"Error in dividing data: {e}")
            raise e

class DataTokenizingStrategy(DataStrategy):
    def handle_data(self, data: Dataset, tokenizer) -> Dataset:
        try:
            tokenizer.pad_token = tokenizer.eos_token
            tokenized_inputs = tokenizer(data["text"], padding="max_length", truncation=True, return_tensors="pt")
            tokenized_targets = tokenizer(data["code"], padding="max_length", truncation=True, return_tensors="pt")
            
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