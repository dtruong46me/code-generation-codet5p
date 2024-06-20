import torch

import os
import sys

import argparse

from transformers import T5ForConditionalGeneration
from peft import IA3Config, get_peft_model, TaskType

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)
from codet5p import FineTunedCodet5Model

class IA3Model(FineTunedCodet5Model):
    def __init__(self, checkpoint: str, args: argparse.Namespace):
        super().__init__(checkpoint, args)

        self.ia3_config = IA3Config(task_type=TaskType.SEQ_2_SEQ_LM,
                                    target_modules=args.target_modules.split(","),
                                    feedforward_modules=["wi","wo"])
    
    def get_ia3_model(self):
        print(f"Get IA3 model")
        return T5ForConditionalGeneration.from_pretrained(self.checkpoint).to(self.device)
    
    def get_peft(self, model, config):
        print(f"Get PEFT model")
        return get_peft_model(model, config)
    
    def get_trainable_parameters(self) -> None:
        print("=================")
        self.origin_model.print_trainable_parameters()
        print("=================")


# Load IA3 model
def load_ia3_model(checkpoint: str, args: argparse.Namespace):
    return IA3Model(checkpoint, args)
