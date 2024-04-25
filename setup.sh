echo "Dinh Truong xin chao anh em!"
echo "Setting up the environment!"
echo "=========================="
pip install -q -U datasets
pip install -q -r requirements.txt
pip install -q -r /content/code-generation-codet5p/requirements.txt
pip install -q -r /kaggle/working/code-generation-codet5p/requirements.txt
echo "Installed packages:\ndatasets\nwandb\nhuggingface_hub\naccelerate\npeft\nbitsandbytes\n"
echo "=========================="
echo "Set up complete!"