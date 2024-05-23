import torch

import os
import sys

import argparse

from transformers import GenerationConfig, T5ForConditionalGeneration
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)
from codet5p import FineTunedCodet5Model

class LoraCodet5p(FineTunedCodet5Model):
    def __init__(self, checkpoint: str, args: argparse.Namespace):
        super().__init__(checkpoint)

        self.lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.lora_model = None
        self.generation_config = GenerationConfig(max_new_tokens=200, temperature=0.7, top_p=0.7)

    def get_peft(self, model, config):
        print(f"Get PEFT model")
        return get_peft_model(model, config)
    
    def get_trainable_parameters(self) -> None:
        print("=================")
        self.lora_model.print_trainable_parameters()
        print("=================")


# Load Lora model
def load_lora_model(checkpoint: str, args: argparse.Namespace):
    try:
        return LoraCodet5p(checkpoint, args)

    except Exception as e:
        print(f"Error while loading QLoRA model: {e}")
        raise e