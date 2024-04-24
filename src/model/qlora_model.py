import logging
import torch
from codet5 import FineTunedCodet5Model

from transformers import BitsAndBytesConfig, GenerationConfig, T5ForConditionalGeneration
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLoraCodet5p(FineTunedCodet5Model):
    def __init__(self, checkpoint):
        super().__init__(checkpoint)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.qlora_model = None
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.5,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.generation_config = GenerationConfig(max_new_tokens=200, temperature=0.7, top_p=0.7)

    def get_qlora_model(self, **kwargs):
        return T5ForConditionalGeneration.from_pretrained(self.checkpoint, quantization_config=self.bnb_config, **kwargs)
    
    def get_peft(self, model, config):
        return get_peft_model(model, config)
    
    def get_trainable_parameters(self):
        return self.qlora_model.print_trainable_parameters()
  
    
# Load Qlora model
def load_qlora_model(checkpoint):
    try:
        return QLoraCodet5p(checkpoint)

    except Exception as e:
        logger.error(f"Error while loading QLoRA model: {e}")
        raise e
    
if __name__=='__main__':
    checkpoint = "Salesforce/codet5p-770m"
    model = QLoraCodet5p(checkpoint)
    model.qlora_model = model.get_qlora_model()

    trainable_params = model.get_trainable_parameters()
    print("Trainable Parameters: ", trainable_params)