import logging
import torch

import os
import sys

import argparse

from transformers import BitsAndBytesConfig, GenerationConfig, T5ForConditionalGeneration
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)
from codet5p import FineTunedCodet5Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLoraCodet5p(FineTunedCodet5Model):
    def __init__(self, checkpoint: str, args: argparse.Namespace):
        super().__init__(checkpoint)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.qlora_model = None
        self.lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.generation_config = GenerationConfig(max_new_tokens=200, temperature=0.7, top_p=0.7)

    def get_qlora_model(self, **kwargs):
        logger.info(f"Get QLoRA model")
        return T5ForConditionalGeneration.from_pretrained(self.checkpoint, quantization_config=self.bnb_config, **kwargs)
    
    def get_peft(self, model, config):
        logger.info(f"Get PEFT model")
        return get_peft_model(model, config)
    
    def get_trainable_parameters(self) -> None:
        print("=================")
        self.qlora_model.print_trainable_parameters()
        print("=================")

  
# Load Qlora model
def load_qlora_model(checkpoint: str, args: argparse.Namespace):
    try:
        return QLoraCodet5p(checkpoint, args)

    except Exception as e:
        logger.error(f"Error while loading QLoRA model: {e}")
        raise e
    
# if __name__=='__main__':
#     args = argparse.ArgumentParser()
#     args.add_argument("--checkpoint", type=str, default="Salesforce/codet5p-770m")
#     args = args.parse_args()
#     # checkpoint = "Salesforce/codet5p-770m"
#     model = QLoraCodet5p(args.checkpoint)

#     model.qlora_model = model.get_qlora_model()

#     model.qlora_model = model.get_peft(model=model.qlora_model, config=model.lora_config)
#     model.get_trainable_parameters()
#     model.origin_model = model.get_codet5p()
#     print(model.origin_model.base_model)