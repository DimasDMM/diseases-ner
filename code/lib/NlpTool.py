import os
import re
import copy
import numpy as np

import string
import en_core_web_sm
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS as STOP_WORDS_SET

class NlpTool:
    MAX_SEQUENCE_LENGTH = 50
    
    def get_nlp_engine(self, basic_tokenizer=False):
        if basic_tokenizer:
            nlp = English()
        else:
            nlp = en_core_web_sm.load()
        return nlp

    def tokenize_texts(self, texts, basic_tokenizer=False):
        texts = texts.copy()
        nlp = self.get_nlp_engine(basic_tokenizer=basic_tokenizer)
        
        for index, text in enumerate(texts):
            nlp_tokens = nlp(text)
            
            pos = [w.pos_ for w in nlp_tokens]
            lemmas = [w.lemma_.strip() if w.lemma_ != "-PRON-" else w.lower_.strip() for w in nlp_tokens]
            lemmas = [w if any(c.isupper() for c in w) else w for w in lemmas]
            
            # Remove empty tokens
            tokens = [(lemmas[i].strip(), pos[i], nlp_tokens[i]) for i in range(len(nlp_tokens)) if lemmas[i].strip() != '']
            
            texts[index] = tokens
        
        return texts
