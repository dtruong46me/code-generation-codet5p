output_path=preds/codet5p-220m-py_T0.2_N200

echo 'Output path: '$output_path
python /kaggle/working/code-generation-codet5p/src/evaluation/process_preds.py --path ${output_path} --out_path ${output_path}.jsonl

evaluate_functional_correctness ${output_path}.jsonl