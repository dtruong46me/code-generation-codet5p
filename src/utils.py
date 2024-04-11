import yaml
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

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
        # eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )
    return trainer


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_tokens(token_path):
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            tokens = yaml.safe_load(f)
            return tokens
        
    else:
        return