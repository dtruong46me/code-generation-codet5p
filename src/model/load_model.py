import logging
from transformers import AutoModelForSeq2SeqLM

class FineTunedCodet5Model:
    def __init__(self, checkpoint, configs, trainer, training_args):
        self.configs = configs
        self.codet5 = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.trainer = trainer
        self.training_args = training_args

    def forward(self):
        pass

    def generate(self, inputs):
        pass