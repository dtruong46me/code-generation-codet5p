#!/bin/bash

echo "[+] Installing packages..."
start_time=$(date +%s)

pip install -q -U datasets
if [[ -f "requirements.txt" ]]; then
    pip install -q -r requirements.txt
fi

if [[ -f "/content/code-generation-codet5p/requirements.txt" ]]; then
    pip install -q -r /content/code-generation-codet5p/requirements.txt
fi

if [[ -f "/kaggle/working/code-generation-codet5p/requirements.txt" ]]; then
    pip install -q -r /kaggle/working/code-generation-codet5p/requirements.txt
fi

echo ""
echo "[+] Finish installing packages"

end_time=$(date +%s)
setup_time=$((end_time - start_time))

mins=$((setup_time / 60))
secs=$((setup_time % 60))

echo "Execution time: $mins mins $secs secs."
