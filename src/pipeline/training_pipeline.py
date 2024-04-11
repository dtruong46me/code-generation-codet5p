import logging

import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)
from utils import *

from model.codet5 import load_model
from data.data_cleaning import clean_data
from data.load_data import ingest_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def training_pipeline(args: argparse.Namespace):
    try:
        # Load model from checkpoint
        model = load_model(args.checkpoint)
        logger.info("Complete loading model!")

        # Load dataset from datapath
        data = ingest_data(args.datapath)
        logger.info("Complete loading dataset!")

        # Clean dataset with strategy
        data = clean_data(data, model.tokenizer)
        logger.info("Complete cleaning dataset!")

        # Load training arguments
        training_args = load_training_arguments(args)
        logger.info("Complete loading training arguments!")

        # Load trainer
        trainer = load_trainer(model=model.codet5, 
                               training_args=training_args, 
                               dataset=data, 
                               tokenizer=model.tokenizer)
        logger.info("Complete loading trainer!")

        # Train model
        trainer.train()
        logger.info("Complete training!")

        # Push trainer to Huggingface Hub
        trainer.push_to_hub()
        logger.info("Complete pushing model to hub!")

    except Exception as e:
        logger.error("Error while training: {e}")
        raise

# if __name__=='__main__':
#     checkpoint = "Salesforce/codet5-base"
#     datapath = "mbpp"
#     configpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"))
#     print(configpath)

#     training_pipeline(checkpoint, datapath, configpath)