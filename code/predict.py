import logging
import numpy as np
import os
import sys

from lib.FileManager import FileManager
from lib.DataProcess import DataProcess
from lib.Jaccard import Jaccard
from lib.ModelFactory import ModelFactory
from lib.LstmCrfModel import LstmCrfModel
from lib.DictionaryModel import DictionaryModel

def run(logging, model_name, text):
    file_manager = FileManager()
    jaccard = Jaccard()
    data_process = DataProcess(jaccard)

    # Create the desired model and train it
    model_factory = ModelFactory(file_manager)
    model_manager = model_factory.make(model_name)
    
    dataset = [text]
    tok_dataset = data_process.tokenize_texts(dataset)

    if isinstance(model_manager, LstmCrfModel):
        # Model: LSTM-CRF
        logging.info('Loading model and vocabularies...')

        (_, word2id, vocab_words_size), (_, pos2id, vocab_pos_size) = model_manager.load_vocabularies(model_name)
        
        logging.info('Encoding text...')
        words = data_process.get_texts_words(tok_dataset)
        pos = data_process.get_texts_pos(tok_dataset)

        words_enc = data_process.encode_texts(words, word2id)
        pos_enc = data_process.encode_texts(pos, pos2id)

        words_enc = data_process.to_sequences(words_enc, DataProcess.MAX_SEQUENCE_LENGTH)
        pos_enc = data_process.to_sequences(pos_enc, DataProcess.MAX_SEQUENCE_LENGTH)
        
        logging.info('Loading model...')
        model = model_manager.create_model(vocab_words_size, vocab_pos_size, DataProcess.MAX_SEQUENCE_LENGTH)
        model_manager.load_model_weights(model_name, model)
        
        logging.info('Finding entities...')
        pred_indicators = model_manager.make_bin_prediction(model, words_enc, pos_enc)
        entities = data_process.bin_to_str(tok_dataset, pred_indicators)      
        
    elif isinstance(model_manager, DictionaryModel):
        # Model: Dictionary
        logging.info('Loading dictionary...')
        diseases = file_manager.get_diseases_dictionary()
        diseases = data_process.tokenize_texts(diseases)
        model_manager.build_dictionary(diseases)
        
        logging.info('Finding entities...')
        _, pred_indicators = model_manager.find_entities(tok_dataset, jaccard.word_jaccard)
        entities = data_process.bin_to_str(tok_dataset, pred_indicators)
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
