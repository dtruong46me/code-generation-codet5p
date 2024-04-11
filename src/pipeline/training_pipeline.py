import logging

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)
from utils import *

from model.codet5 import load_model
from data.data_cleaning import clean_data
from data.load_data import ingest_data


def training_pipeline(checkpoint, datapath, configpath):
    try:
        model = load_model(checkpoint)
        print("Complete loading model!")

        data = ingest_data(datapath)
        print("Complete loading dataset!")

        data = clean_data(data, model.tokenizer)
        print(data.column_names)
        print("Complete cleaning dataset!")

        config = load_config(configpath)
        logging.info("Complete loading config!")

        training_args = load_training_arguments(config)
        logging.info("Complete loading training arguments!")

        trainer = load_trainer(model=model.codet5, training_args=training_args, dataset=data, tokenizer=model.tokenizer)
        logging.info("Complete loading trainer!")

        trainer.train()

        # trainer.push_to_hub()

    except Exception as e:
        logging.error("Error while training: {e}")
        raise

if __name__=='__main__':
    checkpoint = "Salesforce/codet5-base"
    datapath = "mbpp"
    configpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"))
    print(configpath)

    training_pipeline(checkpoint, datapath, configpath)