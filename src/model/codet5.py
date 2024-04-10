import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration
import yaml


class FineTunedCodet5Model:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.codet5 = T5ForConditionalGeneration.from_pretrained(checkpoint)
    
    def generate(self, input_text):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            input_ids = tokenizer(input_text, return_tensors="pt")['input_ids']
            outputs = self.codet5.generate(input_ids, max_length=256)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            logging.error(f"Error while generating: {e}")
            raise e
        
def load_training_arguments(config):
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["training_args"]["output_dir"],
        overwrite_output_dir=config["training_args"]["overwrite_output_dir"],
        num_train_epochs=config["training_args"]["num_train_epochs"],
        evaluation_strategy=config["training_args"]["evaluation_strategy"],
        logging_strategy=config["training_args"]["logging_strategy"],
        logging_steps=config["training_args"]["logging_steps"],
        per_device_train_batch_size=config["training_args"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training_args"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training_args"]["gradient_accumulation_steps"],
        learning_rate=config["training_args"]["learning_rate"],
        report_to=config["training_args"]["report_to"],
        run_name=config["training_args"]["run_name"]
    )
    return training_args

def load_trainer(model, training_args, dataset, tokenizer):
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )
    return trainer
        
def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint):
    try:
        return FineTunedCodet5Model(checkpoint).codet5
    
    except Exception as e:
        logging.error("Error while loading model: {e}")
        raise e

if __name__=='__main__':
    checkpoint = "Salesforce/codet5p-220m-py"
    model = load_model(checkpoint)

    # prompt = "def print_hello_world():"
    prompt = "Write a Python function that accepts a list of words, and returns a  dictionary where keys are the words and values are the frequencies of  the words in the list. Use a for loop, if clause, and dictionary in  your solution."

    output = model.generate(prompt)
    print()
    print(output)
    print()
    print(type(model))