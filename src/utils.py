import yaml
import os
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer
import argparse

import torch.nn.functional as F

import torch
import numpy as np


# Load Training Arguments
def load_training_arguments(args):
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

        save_total_limit=args.save_total_limit,
        predict_with_generate=args.predict_with_generate,

        push_to_hub=args.push_to_hub,
        report_to=args.report_to,
        run_name=args.run_name
    )
    return training_args

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Text2Code and Code2Text generation losses
        labels = inputs.get("labels")
        lt2c = self.label_smoother(logits, labels)
        lc2t = self.label_smoother(logits.transpose(1,2), labels.transpose(1,2))

        # Text-Code contrastive loss
        cls_embeddings = outputs.encoder_last_hidden_state[:,0,:]
        text_embeddings, code_embeddings = cls_embeddings.chunk(2, dim=0)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=-1)

        batch_size = text_embeddings.size(0)
        temperature = self.model.temperature

        # Text-to-code similarities
        st2c = torch.matmul(text_embeddings, code_embeddings.transpose(0, 1))
        pt2c = F.softmax(st2c / temperature, dim=1)
        yt2c = torch.eye(batch_size).to(st2c.device)
        lt2c_contrastive = F.cross_entropy(st2c / temperature, yt2c.argmax(dim=1))

        # Code-to-text similarities
        sc2t = torch.matmul(code_embeddings, text_embeddings.transpose(0, 1))
        pc2t = F.softmax(sc2t / temperature, dim=1)
        yc2t = torch.eye(batch_size).to(sc2t.device)
        lc2t_contrastive = F.cross_entropy(sc2t / temperature, yc2t.argmax(dim=1))

        # Total loss
        loss = lt2c + lc2t + 0.5 * (lt2c_contrastive + lc2t_contrastive)

        if return_outputs:
            return loss, outputs
        else:
            return loss


def load_trainer(model, training_args, dataset, tokenizer):
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        # callbacks=callbacks
    )
    return trainer

def load_tokens(token_path):
    if os.path.exists(token_path):
        with open(token_path, "r") as f:
            tokens = yaml.safe_load(f)
            return tokens
    else:
        return

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tuning CodeT5 Model")
    parser.add_argument("--huggingface_hub_token", type=str, default="")
    parser.add_argument("--wandb_token", type=str, default="")
    parser.add_argument("--datapath", type=str, default="mbpp")
    parser.add_argument("--checkpoint", type=str, default="Salesforce/codet5p-220m", help="Model checkpoint to use")
    
    parser.add_argument("--output_dir", type=str, default="codet5p-220m-running", help="Output directory for fine-tuned model")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--save_strategy", type=str, default="no")

    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--predict_with_generate", action="store_true")
    
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="codet5p-220m")

    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--ia3", action="store_true")


    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--target_modules", type=str, default="q,k")
    parser.add_argument("--feedforward_modules", type=str, default="wi,wo")
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=512)

    args = parser.parse_args()
    return args

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def freeze_decoder_except_xattn_codegen(model):
    print(f"Params before freezing: {model.num_parameters()} || Trainable parameters: {get_model_size(model)}")

    for param in model.decoder.parameters():
        param.requires_grad = False

    num_decoder_layers = model.decoder.config.num_layers
    for i in range(num_decoder_layers):
        each_decoder_layer = model.decoder.block[i]
        if hasattr(each_decoder_layer, "layer"):
            cross_attention_layer = each_decoder_layer.layer[1]  # The second layer is typically the cross-attention layer
            for param in cross_attention_layer.parameters():
                param.requires_grad = True
            cross_attention_layer.to(torch.float32)
        
        if hasattr(each_decoder_layer, "alpha_xattn"):
            each_decoder_layer.alpha_xattn.requires_grad = True
            
    print(f"Params before freezing: {model.num_parameters()} || Trainable parameters: {get_model_size(model)}")
    