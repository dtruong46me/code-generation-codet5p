from human_eval.data import write_jsonl, read_problems
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

checkpoint = "Salesforce/codet5p-220m"
device = "cuda" if torch.cuda.is_available() else "cpu" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)

def generate_one_completion(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
problems = read_problems()

num_samples_per_task = 1
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)