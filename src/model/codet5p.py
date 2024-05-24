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
                                                         torch_dtype=self.args.torch_type,
                                                         trust_remote_code=True).to(self.device)
        return T5ForConditionalGeneration.from_pretrained(self.checkpoint, torch_type=self.args.torch_type).to(self.device)

    def generate(self, input_text, **kwargs):
        try:
            print(f"Generating output for input: {input_text}")
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            outputs = self.origin_model.generate(input_ids, self.generation_config, **kwargs)
            generated_text = self.tokenizer.decode([token for token in outputs[0] if token != -100], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            print(f"Error while generating: {e}")
            raise e
        
    def get_trainable_parameters(self) -> None:
        print("=================")
        print("Total parameters:", self.origin_model.num_parameters())
        print("=================")
  

# Load model    
def load_model(checkpoint, args):
    try:
        return FineTunedCodet5Model(checkpoint, args)
    
    except Exception as e:
        print(f"Error while loading model: {e}")
        raise e

# if __name__=='__main__':
#     checkpoint = "Salesforce/codet5p-770m"
#     model = QLoraCodet5p(checkpoint)
#     model.qlora_model = model.get_qlora_model()

#     trainable_params = model.get_trainable_parameters()
#     print("Trainable Parameters: ", trainable_params)
    

#     model = load_model(checkpoint)

#     # prompt = "def print_hello_world():"
#     # prompt = "Write a function to get a lucid number smaller than or equal to n."
#     prompt = "Only Write a function to reverse words in a given string."

#     output = model.generate(prompt)
#     print()
#     print(output)
#     print()
#     print(type(model))