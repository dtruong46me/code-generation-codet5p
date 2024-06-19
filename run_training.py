import wandb
from huggingface_hub import login

import warnings
warnings.filterwarnings("ignore")

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.pipeline.training_pipeline import training_pipeline
from src.utils import parse_args


if __name__=='__main__':
    args = parse_args()

    checkpoint = args.checkpoint
    datapath = args.datapath


    huggingface_hub_token = args.huggingface_hub_token
    wandb_token = args.wandb_token

    if wandb_token != "":
        os.environ["WANDB_PROJECT"] = "project2"
    
    login(token=huggingface_hub_token)
    wandb.login(key=wandb_token)

    training_pipeline(args)