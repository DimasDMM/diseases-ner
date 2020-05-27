from .LstmCrfModel import LstmCrfModel
from .DictionaryModel import DictionaryModel

class ModelFactory:
    MODEL_LSTM_CRF = 'lstm-crf'
    MODEL_DICTIONARY = 'dictionary'

    def __init__(self, file_manager):
        self._file_manager = file_manager

    def make(self, name):
        if name == self.MODEL_LSTM_CRF:
            return LstmCrfModel(self._file_manager)
        elif name == self.MODEL_DICTIONARY:
            return DictionaryModel(self._file_manager)
        else:
            raise ValueError(name)
