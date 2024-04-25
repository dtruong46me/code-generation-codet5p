import logging

import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)
from utils import *

from model.codet5p import load_model
from model.qlora_model import load_qlora_model
from data.data_cleaning import clean_data
from data.load_data import ingest_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def training_pipeline(args: argparse.Namespace):
    try:
        # Load model from checkpoint
        if args.useqlora==True:
            model = load_qlora_model(args.checkpoint, args)
            model.qlora_model = model.get_qlora_model()
            model.qlora_model = model.get_peft(model.qlora_model, model.lora_config)
            model.get_trainable_parameters()
        if args.uselora==True:
            model = None
        if args.useqlora==False and args.uselora==False:
            model = load_model(args.checkpoint)
            model.origin_model = model.get_codet5p()
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
        if args.useqlora==True:
            trainer = load_trainer(model=model.qlora_model,
                                   training_args=training_args,
                                   dataset=data,
                                   tokenizer=model.tokenizer)
        if args.uselora==True:
            trainer = None
        if args.useqlora==False and args.uselora==False:
            trainer = load_trainer(model=model.origin_model, 
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