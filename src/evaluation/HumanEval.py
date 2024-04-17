# from human_eval.data import write_jsonl, read_problems
# from transformers import AutoModel, AutoTokenizer
# def generate_one_completion(prompt):

# problems = read_problems()

# num_samples_per_task = 200

# samples = [
#     dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
#     for task_id in problems
#     for _ in range(num_samples_per_task)
# ]
# write_jsonl("samples.jsonl", samples)
import sys
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)
print(path)