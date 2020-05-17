import numpy as np
import re
import string

from spacy.lang.en.stop_words import STOP_WORDS as STOP_WORDS_SET

from .ModelBase import ModelBase

class DictionaryModel(ModelBase):
    def __init__(self, file_manager, nlp_tool):
        super().__init__(file_manager, nlp_tool)

        diseases = self._file_manager.getDiseasesDictionary()
        diseases = self._nlp_tool.tokenize_texts(diseases)
        self._tok_diseases = [[d[0] for d in disease] for disease in diseases]

    def get_maxbin_dataset(self, tokenized_dataset):
        ignored_tokens = list(STOP_WORDS_SET) + list(string.punctuation)

        special_separators = re.compile(r'^[0-9_\-\\/]+$')

        bin_entities_per_text = []

        for tokens in tokenized_dataset:
            len_tokens = len(tokens)

            bin_entities = np.zeros(len_tokens, dtype=int)

            # Find diseases coincidences
            for index, token in enumerate(tokens):
                if token in ignored_tokens or special_separators.match(token):
                    # Ignore certain tokens
                    continue

                for disease_tokens in self._tok_diseases:
                    if token in disease_tokens:
                        bin_entities[index] = 1
                        break

            # Given a token with matches as a disease, if the next token is a "certain separator", then add it
            # as a coincidence as well as the next token to the separator. Idem. with previous tokens.

            # Forward
            for i in range(2, len_tokens):
                if special_separators.match(tokens[i - 1]) and bin_entities[i - 2] == 1:
                    bin_entities[i - 1] = 1
                    bin_entities[i] = 1

            # Backwards
            for i in range(len_tokens - 2):
                if special_separators.match(tokens[i + 1]) and bin_entities[i + 2] == 1:
                    bin_entities[i + 1] = 1
                    bin_entities[i] = 1

            bin_entities_per_text.append(bin_entities)
        
        return bin_entities_per_text

    def get_entities(self, text):
        # Put the previous text inside of a list
        texts = list([text])

        # Tokenize it and get PoS tags
        nlp_text = self._nlp_tool.tokenize_texts(texts)[0]
        
        tok_text = list([[w[0] for w in nlp_text]])
        pred_bin_text = self.get_maxbin_dataset(tok_text)[0]
        
        pred_str_entities = self.bin_pred_to_str(nlp_text, pred_bin_text)
        
        return pred_str_entities
