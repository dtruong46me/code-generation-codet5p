import wandb
from huggingface_hub import notebook_login

import warnings
warnings.filterwarnings("ignore")

import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, path)

from src.pipeline.training_pipeline import training_pipeline


if __name__=='__main__':
    notebook_login()
    wandb.login()

    checkpoint = "Salesforce/codet5-base"
    datapath = "mbpp"
    configpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.yaml"))

    training_pipeline(checkpoint, datapath, configpath)