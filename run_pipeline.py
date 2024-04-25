import wandb
from huggingface_hub import login

import warnings
warnings.filterwarnings("ignore")

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.pipeline.training_pipeline import training_pipeline
from src.utils import load_tokens, parse_args


if __name__=='__main__':
    # checkpoint = "Salesforce/codet5-base"
    # datapath = "mbpp"
    # configpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parse_args()

    checkpoint = args.checkpoint
    datapath = args.datapath
    configpath = args.configpath

    #token_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "credential.yaml"))

    #tokens = load_tokens(token_path)

    huggingface_hub_token = args.huggingface_hub_token
    wandb_token = args.wandb_token


    if huggingface_hub_token:
        os.environ["HUGGINGFACE_TOKEN"] = huggingface_hub_token

    if wandb_token != "":
        os.environ["WANDB_PROJECT"] = "code_generation"
        os.environ["WANDB_API_KEY"] = wandb_token
 
    
    login(token=huggingface_hub_token)
#     wandb.login(key=wandb_token)

    training_pipeline(args)