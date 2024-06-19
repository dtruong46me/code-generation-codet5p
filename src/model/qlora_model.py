import torch

import os
import sys

import argparse

from transformers import BitsAndBytesConfig, GenerationConfig, T5ForConditionalGeneration
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)
from lora_model import LoraCodet5p

class QLoraCodet5p(LoraCodet5p):
    def __init__(self, checkpoint: str, args: argparse.Namespace):
        super().__init__(checkpoint, args)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def get_qlora_model(self, **kwargs):
        print(f"Get QLoRA model")
        return T5ForConditionalGeneration.from_pretrained(self.checkpoint, quantization_config=self.bnb_config, **kwargs)
    
    def get_trainable_parameters(self) -> None:
        print("=================")
        self.origin_model.print_trainable_parameters()
        print("=================")

  
# Load Qlora model
def load_qlora_model(checkpoint: str, args: argparse.Namespace):
    return QLoraCodet5p(checkpoint, args)