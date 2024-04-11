import wandb
from huggingface_hub import notebook_login

import warnings
warnings.filterwarnings("ignore")

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.pipeline.training_pipeline import training_pipeline
from src.utils import load_tokens


if __name__=='__main__':
    checkpoint = "Salesforce/codet5-base"
    datapath = "mbpp"
    configpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.yaml"))
    token_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "credential.yaml"))

    tokens = load_tokens(token_path)

    if tokens:
        huggingface_hub_token = tokens["huggingface_hub_token"]
        wandb_token = tokens["wandb_token"]

        if huggingface_hub_token:
            os.environ["HUGGINGFACE_TOKEN"] = huggingface_hub_token

        if wandb_token:
            os.environ["WANDB_API_KEY"] = wandb_token
    
    notebook_login(huggingface_hub_token)
    wandb.login(wandb_token)

    training_pipeline(checkpoint, datapath, configpath)