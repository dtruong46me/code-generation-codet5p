from human_eval.data import write_jsonl, read_problems
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import argparse

def main():
    # Argument Parsers
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Salesforce/codet5p-220m-py', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=600, help="")
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--overwrite', action='store_true', help='')
    args = parser.parse_args()

    # Load model and tokenizer
    checkpoint = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu" # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

    # Generate completions
    def generate_one_completion(prompt):
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(inputs, 
                                 max_length=args.max_len,
                                 do_sample=True,
                                 top_p=args.top_p,
                                 temperature=args.temperature)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    problems = read_problems()

    # Generate completions for each task
    num_samples_per_task = args.N
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]

    # Write completions to file
    write_jsonl(args.output_path, samples)

if __name__ == "__main__":
    main()
