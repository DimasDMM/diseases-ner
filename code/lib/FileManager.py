import os
import pickle

class FileManager:
    DATA_PATH = '../../data/'
    ARTIFACTS_PATH = '../../artifacts/'

    RAW_TRAIN_FILE = 'NCBI_corpus_training.txt'
    RAW_TEST_FILE = 'NCBI_corpus_testing.txt'

    DISEASES_FILE = 'diseases.txt'
    
    def __init__(self):
        self.DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/' + self.DATA_PATH
        self.ARTIFACTS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/' + self.ARTIFACTS_PATH
    
    def get_diseases_dictionary(self):
        with open(self.DATA_PATH + self.DISEASES_FILE, 'r') as fp:
            diseases = fp.readlines()
        return diseases

    def get_dataset(self):
        with open(self.DATA_PATH + self.RAW_TRAIN_FILE, 'r') as fp:
            train_dataset = fp.readlines()

        with open(self.DATA_PATH + self.RAW_TEST_FILE, 'r') as fp:
            test_dataset = fp.readlines()
        
        return train_dataset, test_dataset

    def load_model_weights(self, model_name, model):
        model_path = self._get_model_path(model_name)
        model.load_weights(model_path + 'model.h5')
        return model
    
    def save_model_weights(self, model_name, model):
        model_path = self._get_model_path(model_name)
        self._create_model_path(model_name)

        model.save(model_path + 'model.h5')
    
    def load_vocabulary(self, model_name, vocabulary_name):
        self._create_model_path(model_name)
        model_path = self._get_model_path(model_name)
        
        with open(model_path + vocabulary_name + '.pickle', 'rb') as fp:
            tokenizer = pickle.load(fp)
        
        return tokenizer
    
    def save_vocabulary(self, model_name, vocabulary_name, vocabulary):
        self._create_model_path(model_name)
        model_path = self._get_model_path(model_name)
        with open(model_path + vocabulary_name + '.pickle', 'wb') as fp:
            pickle.dump(vocabulary, fp)

    def _create_model_path(self, model_name):
        model_path = self._get_model_path(model_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
    
    def _get_model_path(self, model_name):
        model_path = self.ARTIFACTS_PATH + model_name + '/'
        return model_path
