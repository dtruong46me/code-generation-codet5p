import logging
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

logger = logging.getLogger(__name__)

class FineTunedCodet5Model:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.codet5 = T5ForConditionalGeneration.from_pretrained(checkpoint).to(self.device)
    
    def generate(self, input_text):
        try:
            logger.info("Generating output for input: {input_text}")
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            outputs = self.codet5.generate(input_ids, max_length=1024)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            logger.error(f"Error while generating: {e}")
            raise e
        
        
def load_model(checkpoint):
    try:
        return FineTunedCodet5Model(checkpoint)
    
    except Exception as e:
        logger.error("Error while loading model: {e}")
        raise e

# if __name__=='__main__':
#     checkpoint = "Salesforce/codet5p-220m-py"
#     model = load_model(checkpoint)

#     # prompt = "def print_hello_world():"
#     # prompt = "Write a function to get a lucid number smaller than or equal to n."
#     prompt = "Only Write a function to reverse words in a given string."

#     output = model.generate(prompt)
#     print()
#     print(output)
#     print()
#     print(type(model))