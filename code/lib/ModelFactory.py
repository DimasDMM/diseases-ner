from .LstmCrfModel import LstmCrfModel
from .DictionaryModel import DictionaryModel

class ModelFactory:
    MODEL_LSTM_CRF = 'lstm-crf'
    MODEL_DICTIONARY = 'dictionary'

    def __init__(self, file_manager, nlp_tool):
        self._file_manager = file_manager
        self._nlp_tool = nlp_tool

    def make(self, name):
        if name == self.MODEL_LSTM_CRF:
            return LstmCrfModel(self._file_manager, self._nlp_tool)
        elif name == self.MODEL_DICTIONARY:
            return DictionaryModel(self._file_manager, self._nlp_tool)
        else:
            raise ValueError(name)
