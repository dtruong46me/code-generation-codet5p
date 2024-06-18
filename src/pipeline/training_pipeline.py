import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)
from utils import *

from model.codet5p import load_model
from model.qlora_model import load_qlora_model
from model.lora_model import load_lora_model
from data.data_cleaning import clean_data
from data.load_data import ingest_data


def training_pipeline(args: argparse.Namespace):
    try:
        print("=========================================")
        print('\n'.join(f' + {k}={v}' for k, v in vars(args).items()))
        print("=========================================")

        # Load model from checkpoint
        if args.lora==False:
            model = load_model(args.checkpoint, args)
            model.origin_model = model.get_codet5p()
            model.get_trainable_parameters()
        
        if args.lora==True:
            if args.quantization==False:
                model = load_lora_model(args.checkpoint, args)
                model.origin_model = model.get_lora_model()
                model.origin_model = model.get_peft(model.origin_model, model.lora_config)
                model.get_trainable_parameters()
            
            if args.quantization==True:
                model = load_qlora_model(args.checkpoint, args)
                model.origin_model = model.get_qlora_model()
                model.origin_model = model.get_peft(model.origin_model, model.lora_config)
                model.get_trainable_parameters()

        print("[+] Complete loading model!")

        # Load dataset from datapath
        data = ingest_data(args.datapath)
        print("[+] Complete loading dataset!")

        # Clean dataset with strategy
        data = clean_data(data, model.tokenizer)
        print("[+] Complete cleaning dataset!")

        # Load training arguments
        training_args = load_training_arguments(args)
        print("[+] Complete loading training arguments!")

        # Load trainer
        trainer = load_trainer(model=model.origin_model, 
                            training_args=training_args, 
                            dataset=data, 
                            tokenizer=model.tokenizer)
        print("[+] Complete loading trainer!")

        # Train model
        trainer.train()
        print("[+] Complete training!")

        # Push trainer to Huggingface Hub
        trainer.push_to_hub()
        print("[+] Complete pushing model to hub!")

    except Exception as e:
        print(f"Error while training: {e}")
        raise e