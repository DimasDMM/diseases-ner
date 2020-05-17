import logging
import numpy as np

from lib.FileManager import FileManager
from lib.NlpTool import NlpTool
from lib.ModelFactory import ModelFactory
from lib.LstmCrfModel import LstmCrfModel

EPOCHS = 5

def run(logging, model_name):
    file_manager = FileManager()
    nlp_tool = NlpTool()

    # Use the processed data to train the model
    logging.info('Loading dataset...')
    train_dataset, test_dataset, bin_train_dataset, _ = file_manager.getParsedDataset()

    bin_train_dataset = [s.strip().split(' ') if s.strip() != '' else [] for s in bin_train_dataset]
    bin_train_dataset = [np.array(s).astype('int') for s in bin_train_dataset]

    logging.info('Tokenizing dataset...')
    train_dataset = nlp_tool.tokenize_texts(train_dataset)
    test_dataset = nlp_tool.tokenize_texts(test_dataset)

    # Create the desired model and train it
    model_factory = ModelFactory(file_manager, nlp_tool)
    model_manager = model_factory.make(model_name)

    if isinstance(model_manager, LstmCrfModel):
        # Model: LSTM-CRF
        logging.info('Creating model and keras tokenizers...')

        full_dataset = train_dataset + test_dataset
        word_tokenizer, pos_tokenizer = model_manager.create_keras_tokenizers(full_dataset)
        
        model = model_manager.create_model(word_tokenizer, pos_tokenizer, NlpTool.MAX_SEQUENCE_LENGTH)
        model_manager.train_model(model, word_tokenizer, pos_tokenizer, train_dataset, bin_train_dataset)

        model_manager.save_model_weights(model_name, model)
        model_manager.save_tokenizers(model_name, word_tokenizer, pos_tokenizer)
    else:
        pass # ... Add other trainable models here
    
    logging.info('Done!')

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info('## TRAIN MODEL ##')
    run(logging, ModelFactory.MODEL_LSTM_CRF)
