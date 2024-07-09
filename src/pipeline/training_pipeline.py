import os
import sys
import argparse

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)
from utils import *

from model.codet5p import load_model
from model.qlora_model import load_qlora_model
from model.lora_model import load_lora_model
from model.ia3_model import load_ia3_model
from data.preprocessing import split_data_into_train_valid, tokenize_data
from data.load_data import ingest_data
from transformers import Seq2SeqTrainer

def training_pipeline(args: argparse.Namespace) -> None:
    try:
        print("=========================================")
        print('\n'.join(f' + {k}={v}' for k, v in vars(args).items()))
        print("=========================================")

        # Supervised fine tuning (sft)
        if args.fine_tuning=="sft":
            model = load_model(args.checkpoint, args)
            model.origin_model = model.get_codet5p()
        
        # PEFT - LoRA
        if args.fine_tuning=="lora":
            model = load_lora_model(args.checkpoint, args)
            model.origin_model = model.get_lora_model()
            model.origin_model = model.get_peft(model.origin_model, model.lora_config)
        
        # PEFT - QLoRA
        if args.fine_tuning=="qlora":
            model = load_qlora_model(args.checkpoint, args)
            model.origin_model = model.get_qlora_model()
            model.origin_model = model.get_peft(model.origin_model, model.qlora_config)
        
        # PEFT - IA3
        if args.fine_tuning=="ia3":
            model = load_ia3_model(args.checkpoint, args)
            model.origin_model = model.get_ia3_model()
            model.origin_model = model.get_peft(model.origin_model, model.ia3_config)

        model.get_trainable_parameters()
        print("[+] Complete loading model!")

        # Freeze decoder parameters (exept cross attention)
        freeze_decoder_except_xattn_codegen(model.origin_model)
        print("[+] Freeze decoder parameters except cross attention!")

        # Load dataset from datapath
        dataset = ingest_data(args.datapath, split="train")
        train_data, valid_data = split_data_into_train_valid(dataset, test_size=0.2)
        print("[+] Complete loading dataset!")

        # Clean dataset with strategy
        train_data = tokenize_data(train_data, model.tokenizer)
        valid_data = tokenize_data(valid_data, model.tokenizer)
        print("[+] Complete cleaning dataset!")

        # Load training arguments
        training_args = load_training_arguments(args)
        print("[+] Complete loading training arguments!")

        # Load trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=valid_data,
            tokenizer=model.tokenizer
        )

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