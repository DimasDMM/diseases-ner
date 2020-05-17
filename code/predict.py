import logging
import numpy as np
import os
import sys

from keras.utils import to_categorical
import tensorflow as tf

from lib.FileManager import FileManager
from lib.NlpTool import NlpTool
from lib.ModelFactory import ModelFactory
from lib.LstmCrfModel import LstmCrfModel
from lib.DictionaryModel import DictionaryModel

EPOCHS = 5

def run(logging, model_name, text):
    file_manager = FileManager()
    nlp_tool = NlpTool()

    # Create the desired model and train it
    model_factory = ModelFactory(file_manager, nlp_tool)
    model_manager = model_factory.make(model_name)

    if isinstance(model_manager, LstmCrfModel):
        # Model: LSTM-CRF
        logging.info('Loading model and keras tokenizers...')

        word_tokenizer, pos_tokenizer = model_manager.load_tokenizers(model_name)
        
        model = model_manager.create_model(word_tokenizer, pos_tokenizer, NlpTool.MAX_SEQUENCE_LENGTH)
        model = model_manager.load_model_weights(model_name, model)

        logging.info('Finding entities...')
        entities = model_manager.get_entities(text, model, word_tokenizer, pos_tokenizer)
    elif isinstance(model_manager, DictionaryModel):
        # Model: Dictionary
        logging.info('Finding entities...')
        entities = model_manager.get_entities(text)
    else:
        pass # ... Add other models here
    
    return entities

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info('## TEXT PREDICTION ##')
    
    if len(sys.argv) != 3:
        raise Exception('Please, provide the name of a model and a text.')
    
    entities = run(logging, sys.argv[1], sys.argv[2])
    for e in entities:
        print(e)
