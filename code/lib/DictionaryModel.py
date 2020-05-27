import numpy as np
from .ModelBase import ModelBase

class DictionaryModel(ModelBase):
    def __init__(self, file_manager):
        super().__init__(file_manager)

    def build_dictionary(self, tokenized_dict_diseases):
        self.tokenized_dict_diseases = tokenized_dict_diseases

    def find_entities(self, tok_dataset, jaccard_fn, min_score=0.7):
        entities_per_text = []
        indicators_per_text = []
        
        for text in tok_dataset:
            text_len = len(text)
            t_entities = []
            t_indicator = []
            
            i = 0
            while i < text_len:
                entity_found = False
                
                for entity in self.tokenized_dict_diseases:
                    entity_len = len(entity)
                    score = 0
                    
                    if entity_len + i > text_len:
                        # The entity cannot fit in the tokenized words
                        continue

                    k = 0
                    while k < entity_len:
                        score = score + jaccard_fn(entity[k][2].text, text[i + k][2].text)
                        k = k + 1
                    
                    score = score / entity_len
                    if score >= min_score:
                        entity_found = True
                        t_entities.append(text[i:i+k])
                        t_indicator = t_indicator + [1]*k
                        i = i + k
                
                if not entity_found:
                    t_indicator.append(0)
                    i = i + 1
            
            entities_per_text.append(np.array(t_entities))
            indicators_per_text.append(np.array(t_indicator))
        
        return np.array(entities_per_text), np.array(indicators_per_text)
