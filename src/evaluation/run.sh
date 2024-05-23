model="codet5p-220m-py"
output_path="src/evaluation/preds/codet5p-220m-py_T0.2_N200"
temp=0.8
N=5
max_len=800
num_seqs_per_iter=25

python src/evaluation/generate_code.py --model Salesforce/${model} \
    --temperature ${temp} \
    --num_seqs_per_iter ${num_seqs_per_iter} --N ${N} --max_len ${max_len} --output_path ${output_path} --overwrite