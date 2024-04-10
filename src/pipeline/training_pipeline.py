
from data.data_cleaning import *
from data.data_strategy import *
from data.load_data import *
from evaluation.evaluation import Evaluation

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, AutoModelForSeq2SeqLM

def train_pipeline(data_path, checkpoint="Salesforce/codet5-220m"):
    data = ingest_data(data_path)

    train_ds, test_ds, valid_ds = clean_data(data)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    loaded_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    training_args = Seq2SeqTrainingArguments(
        output_dir="codet5-mbpp-220m",
        overwrite_output_dir=True,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        push_to_hub=False,
        learning_rate=5e-5,
        report_to="wandb",
        run_name="codet5-mbpp-220m"
    )

    trainer = Seq2SeqTrainer(
        model=loaded_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer
    )

    trainer.train()

    evaluation = Evaluation()