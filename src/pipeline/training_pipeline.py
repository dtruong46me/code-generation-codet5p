
from data.data_cleaning import *
from data.data_strategy import *
from data.load_data import *

def train_pipeline(data_path, program_language):
    data = ingest_data(data_path, program_language)

    train_ds, test_ds, valid_ds = clean_data(data)