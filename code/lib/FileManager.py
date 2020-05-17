import os
import pickle

class FileManager:
    DATA_PATH = '../../data/'
    PARSED_DATA_PATH = DATA_PATH + 'parsed/'
    ARTIFACTS_PATH = '../../artifacts/'

    RAW_TRAIN_FILE = 'NCBI_corpus_training.txt'
    RAW_TEST_FILE = 'NCBI_corpus_testing.txt'

    PARSED_TRAIN_FILE = 'corpus_train.txt'
    PARSED_TEST_FILE = 'corpus_test.txt'
    BIN_PARSED_TRAIN_FILE = 'bin_corpus_train.txt'
    BIN_PARSED_TEST_FILE = 'bin_corpus_test.txt'

    DISEASES_FILE = 'diseases.txt'

    NEW_LINE_CHAR = "\n"
    
    def __init__(self):
        self.DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/' + self.DATA_PATH
        self.PARSED_DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/' + self.PARSED_DATA_PATH
        self.ARTIFACTS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/' + self.ARTIFACTS_PATH
    
    def getDiseasesDictionary(self):
        with open(self.DATA_PATH + self.DISEASES_FILE, 'r') as fp:
            diseases = fp.readlines()
        return diseases

    def getRawDataset(self):
        with open(self.DATA_PATH + self.RAW_TRAIN_FILE, 'r') as fp:
            train_dataset = fp.readlines()

        with open(self.DATA_PATH + self.RAW_TEST_FILE, 'r') as fp:
            test_dataset = fp.readlines()
        
        return train_dataset, test_dataset

    def getParsedDataset(self):
        with open(self.PARSED_DATA_PATH + self.PARSED_TRAIN_FILE, 'r') as fp:
            train_dataset = fp.readlines()

        with open(self.PARSED_DATA_PATH + self.PARSED_TEST_FILE, 'r') as fp:
            test_dataset = fp.readlines()

        with open(self.PARSED_DATA_PATH + self.BIN_PARSED_TRAIN_FILE, 'r') as fp:
            bin_train_dataset = fp.readlines()

        with open(self.PARSED_DATA_PATH + self.BIN_PARSED_TEST_FILE, 'r') as fp:
            bin_test_dataset = fp.readlines()
        
        return train_dataset, test_dataset, bin_train_dataset, bin_test_dataset

    def saveParsedDataset(self, train_dataset, test_dataset, bin_train_dataset, bin_test_dataset):
        if not os.path.exists(self.PARSED_DATA_PATH):
            os.mkdir(self.PARSED_DATA_PATH)

        with open(self.PARSED_DATA_PATH + self.PARSED_TRAIN_FILE, 'w') as fp:
            fp.write(self.NEW_LINE_CHAR.join(train_dataset))

        with open(self.PARSED_DATA_PATH + self.PARSED_TEST_FILE, 'w') as fp:
            fp.write(self.NEW_LINE_CHAR.join(test_dataset))

        with open(self.PARSED_DATA_PATH + self.BIN_PARSED_TRAIN_FILE, 'w') as fp:
            fp.write(self.NEW_LINE_CHAR.join(bin_train_dataset))

        with open(self.PARSED_DATA_PATH + self.BIN_PARSED_TEST_FILE, 'w') as fp:
            fp.write(self.NEW_LINE_CHAR.join(bin_test_dataset))

    def load_model_weights(self, model_name, model):
        model_path = self._get_model_path(model_name)
        model.load_weights(model_path + 'model.h5')
        return model
    
    def save_model_weights(self, model_name, model):
        model_path = self._get_model_path(model_name)
        self._create_model_path(model_name)

        model.save_weights(model_path + 'model.h5')
    
    def load_tokenizer(self, model_name, tokenizer_name):
        self._create_model_path(model_name)
        model_path = self._get_model_path(model_name)
        
        with open(model_path + tokenizer_name + '.pickle', 'rb') as fp:
            tokenizer = pickle.load(fp)
        
        return tokenizer
    
    def save_tokenizer(self, model_name, tokenizer_name, tokenizer):
        self._create_model_path(model_name)
        model_path = self._get_model_path(model_name)
        with open(model_path + tokenizer_name + '.pickle', 'wb') as fp:
            pickle.dump(tokenizer, fp)

    def _create_model_path(self, model_name):
        model_path = self._get_model_path(model_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
    
    def _get_model_path(self, model_name):
        model_path = self.ARTIFACTS_PATH + model_name + '/'
        return model_path
