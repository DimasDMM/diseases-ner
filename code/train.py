import logging
import numpy as np

from lib.FileManager import FileManager
from lib.DataProcess import DataProcess
from lib.Jaccard import Jaccard
from lib.ModelFactory import ModelFactory
from lib.LstmCrfModel import LstmCrfModel

EPOCHS = 5

def run(logging, model_name):
    file_manager = FileManager()
    jaccard = Jaccard()
    data_process = DataProcess(jaccard)
    
    # Create the desired model
    model_factory = ModelFactory(file_manager)
    model_manager = model_factory.make(model_name)

    # Use the processed data to train the model
    logging.info('Loading dataset...')
    train_dataset, test_dataset = file_manager.get_dataset()

    logging.info('Initial cleaning...')
    train_dataset = data_process.apply_initial_cleaner(train_dataset)
    test_dataset = data_process.apply_initial_cleaner(test_dataset)

    logging.info('Tokenizing dataset...')
    tok_train_dataset = data_process.tokenize_texts(train_dataset)
    tok_test_dataset = data_process.tokenize_texts(test_dataset)

    logging.info('Preparing some data')
    train_words = data_process.get_texts_words(tok_train_dataset)
    test_words = data_process.get_texts_words(tok_test_dataset)
    train_indicators = data_process.get_indicator_sequences(tok_train_dataset)
    test_indicators = data_process.get_indicator_sequences(tok_test_dataset)

    if isinstance(model_manager, LstmCrfModel):
        # Model: LSTM-CRF
        logging.info('Building vocabularies...')

        train_pos = data_process.get_texts_pos(tok_train_dataset)
        test_pos = data_process.get_texts_pos(tok_test_dataset)

        additional = [DataProcess.UNK]
        vocab_words, word2id, vocab_words_size = data_process.get_vocab_dictionary(train_words, additional)
        vocab_pos, pos2id, vocab_pos_size = data_process.get_vocab_dictionary(train_pos, additional)
        
        logging.info('Encoding texts...')
        train_words_enc = data_process.encode_texts(train_words, word2id)
        train_pos_enc = data_process.encode_texts(train_pos, pos2id)
        
        train_words_enc = data_process.to_sequences(train_words_enc, DataProcess.MAX_SEQUENCE_LENGTH)
        train_pos_enc = data_process.to_sequences(train_pos_enc, DataProcess.MAX_SEQUENCE_LENGTH)
        train_indicators = data_process.to_sequences(train_indicators, DataProcess.MAX_SEQUENCE_LENGTH)
        
        train_indicators_cat = data_process.to_categorical(train_indicators)
        
        logging.info('Creating model and training...')
        model = model_manager.create_model(vocab_words_size, vocab_pos_size, DataProcess.MAX_SEQUENCE_LENGTH)
        model_manager.train_model(model, vocab_words, vocab_pos, train_words_enc, train_pos_enc, train_indicators_cat)
        
        logging.info('Computing Jaccard Index...')
        test_words_enc = data_process.encode_texts(test_words, word2id)
        test_pos_enc = data_process.encode_texts(test_pos, pos2id)
        test_words_enc = data_process.to_sequences(test_words_enc, DataProcess.MAX_SEQUENCE_LENGTH)
        test_pos_enc = data_process.to_sequences(test_pos_enc, DataProcess.MAX_SEQUENCE_LENGTH)
        test_indicators = data_process.to_sequences(test_indicators, DataProcess.MAX_SEQUENCE_LENGTH)
        
        pred_indicators = model_manager.make_bin_prediction(model, test_words_enc, test_pos_enc)
        jaccard_score = jaccard.bin_jaccard(test_indicators, pred_indicators)
        print('- Score: %.4f' % jaccard_score)

        logging.info('Saving model...')
        model_manager.save_model_weights(model_name, model)
        model_manager.save_vocabularies(model_name, vocab_words, vocab_pos)
    else:
        pass # ... Add other trainable models here
    
    logging.info('Done!')

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info('## TRAIN MODEL ##')
    run(logging, ModelFactory.MODEL_LSTM_CRF)
