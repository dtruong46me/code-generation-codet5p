import logging
from abc import ABC, abstractclassmethod
from datasets import load_dataset
from load_data import IngestDataset

class DataStrategy(ABC):
    """
    Abstract class for handling data
    """
    @abstractclassmethod
    def handle_data(self, data):
        pass


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: IngestDataset):
        try:
            train_ds = data["train"]
            valid_ds = data["validation"]
            test_ds = data["test"]

            return train_ds, valid_ds, test_ds

        except Exception as e:
            logging.error(f"Error in dividing data: {e}")


class DataPreprocessingStrategy(DataStrategy):
    def handle_data(self, data):
        return