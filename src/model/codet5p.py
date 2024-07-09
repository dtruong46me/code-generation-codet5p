import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, BitsAndBytesConfig, GenerationConfig
import torch

class FineTunedCodet5Model:
    def __init__(self, checkpoint, args: argparse.Namespace):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.origin_model = None
        self.args = args

        self.generation_config = GenerationConfig(max_new_tokens=args.max_new_tokens, 
                                                  do_sample=True, 
                                                  temperature=args.temperature, 
                                                  top_k=args.top_k, top_p=args.top_p)
    
    def get_codet5p(self):
        if self.checkpoint=="Salesforce/codet5p-2b" or self.checkpoint=="Salesforce/codet5p-6b":
            return AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint,
                                                         torch_dtype=torch.bfloat16,
                                                         trust_remote_code=True).to(self.device)
        return T5ForConditionalGeneration.from_pretrained(self.checkpoint).to(self.device)

    def generate(self, input_text):
        try:
            print(f"Generating from query: ### {input_text}")
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            outputs = self.origin_model.generate(input_ids, self.generation_config, do_sample=True)
            generated_text = self.tokenizer.decode([token for token in outputs[0] if token != -100], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            print(f"Error while generating: {e}")
            raise e
        
    def get_trainable_parameters(self) -> None:
        print("=========================================")
        print("Total parameters:", self.origin_model.num_parameters())
        print("=========================================")
  

# Load model    
def load_model(checkpoint, args):
    return FineTunedCodet5Model(checkpoint, args)