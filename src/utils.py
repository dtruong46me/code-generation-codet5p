import yaml
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import argparse
import logging
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_arguments(args):
    if args.configpath is not None:
        config = load_config(configpath=args.configpath)
        training_args = Seq2SeqTrainingArguments(**config)
    
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=args.overwrite_output_dir,

            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,

            evaluation_strategy=args.evaluation_strategy,
            save_strategy=args.save_strategy,

            logging_strategy=args.logging_strategy,
            logging_steps=args.logging_steps,

            metric_for_best_model=args.metric_for_best_model,
            save_total_limit=args.save_total_limit,

            push_to_hub=args.push_to_hub,
            report_to=args.report_to,
            run_name=args.run_name
        )
    return training_args


def load_trainer(model, training_args, dataset, tokenizer):
    # callbacks = [WandBCallback(model.tokenizer)]
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        # callbacks=callbacks
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
    parser.add_argument("--datapath", type=str, default="mbpp")
    parser.add_argument("--checkpoint", type=str, default="Salesforce/codet5p-220m", help="Model checkpoint to use")
    
    parser.add_argument("--output_dir", type=str, default="codet5p-220m-running", help="Output directory for fine-tuned model")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=int, default=0.00005)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--save_strategy", type=str, default="no")

    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=1)

    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--load_best_model_at_end", type=bool, default=False)
    
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="codet5p-220m-running")

    parser.add_argument("--uselora", type=bool, default=False)
    parser.add_argument("--useqlora", type=bool, default=False)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--target_modules", type=str, default="q,k")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()
    return args

# if __name__=='__main__':
#     args = parse_args()
#     print(args)
#     print(type(args))
#     print(args.num_train_epochs)