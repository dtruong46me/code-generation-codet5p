
# --input_text: str
# --output_text: str
# --model: Fine tuned Model
# --config: GenerationConfig

import torch

import argparse

from transformers import (
    GenerationConfig,
    T5ForConditionalGeneration,
    AutoTokenizer
)


def inference(model, tokenizer, input_text: str, generation_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = f"""Generate Python code from the following instruction:
### Instruction: {input_text}
### Response:"""
    
    print("\n[+] Input:\n", input_text)
    model.to(device)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, do_sample=True, generation_config=generation_config)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n[+] Output:\n", output_text)
    return output_text


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="dtruong46me/codet5p-770m-2")
    parser.add_argument("--input_text", type=str, default="Print Hello World")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=8)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens
    )

    response = inference(model, tokenizer, args.input_text, generation_config)
    print("\n[+] Response:", response)