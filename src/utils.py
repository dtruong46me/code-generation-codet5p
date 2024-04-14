import yaml
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse

def load_training_arguments(args):
    if args.configpath is not None:
        config = load_config(configpath=args.configpath)
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
    
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=args.overwrite_output_dir,
            num_train_epochs=args.num_train_epochs,
            evaluation_strategy=args.evaluation_strategy,
            logging_strategy=args.logging_strategy,
            logging_steps=args.logging_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            report_to=args.report_to,
            run_name=args.run_name
        )
    return training_args


def load_trainer(model, training_args, dataset, tokenizer):
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer
    )
    return trainer


def load_config(configpath):
    with open(configpath, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_tokens(token_path):
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            tokens = yaml.safe_load(f)
            return tokens
    else:
        return
    
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tuning CodeT5 Model")
    parser.add_argument("--configpath", type=str, default=None, help="Path to the config.yaml")
    parser.add_argument("--huggingface_hub_token", type=str, default="none")
    parser.add_argument("--wandb_token", type=str, default="none")
    parser.add_argument("--checkpoint", type=str, default="Salesforce/codet5-base", help="Model checkpoint to use")
    parser.add_argument("--output_dir", type=str, default="codet5-fine-tuned", help="Output directory for fine-tuned model")
    parser.add_argument("--overwrite_output_dir", type=bool, default=True)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--weight_decay", type=int, default=0.01)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=int, default=0.00005)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="codet5-base-fine-tuned")
    parser.add_argument("--datapath", type=str, default="mbpp")
    args = parser.parse_args()
    return args

# if __name__=='__main__':
#     args = parse_args()
#     print(args)
#     print(type(args))
#     print(args.num_train_epochs)