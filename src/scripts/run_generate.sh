smodel=codet5p-220m-py
temp=0.2
max_len=800
pred_num=200
num_seqs_per_iter=25 # 25 for 350M and 770M, 10 for 2B, 8 for 6B, 2 for 16B on A100-40G

output_path=preds/${model}_T${temp}_N${pred_num}

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems, 21 per GPU if GPU=8
index=0
gpu_num=2
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 82))
  end_index=$(((i + 1) * 82))

    gpu=$((i))
    echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
    ((index++))
    (
        CUDA_VISIBLE_DEVICES=$gpu python /kaggle/working/code-generation-codet5p/src/evaluation/generate_code.py --model Salesforce/${model} \
        --start_index ${start_index} --end_index ${end_index} --temperature ${temp} \
        --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path}
    ) &
    if (($index % $gpu_num == 0)); then wait; fi
done