import re
import numpy as np

class InitialPreprocess:
    MAX_SEQUENCE_LENGTH = 1000
    
    def __init__(self, nlp_tool):
        self._nlp_tool = nlp_tool
    
    def apply_initial_number_remover(self, texts):
        texts = [re.sub(r'^[0-9]+\s*', '', text) for text in texts]
        return texts

    def apply_sentences_split(self, texts):
        nlp = self._nlp_tool.get_nlp_engine()
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        
        sentences = []
        for text in texts:
            t_sentences = nlp(text)
            t_sentences = [s.text.strip() for s in t_sentences.sents if s.text.strip() != '']
            sentences = sentences + t_sentences
        
        return sentences

    def apply_tag_cleaner(self, texts):
        texts = texts.copy()
        for index, text in enumerate(texts):
            text = re.sub(r'<category=".+?">', ' ', text)
            text = re.sub(r'</category>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            texts[index] = text.strip()
        
        return texts

    """
    Returns a list "text ID -> list of entities" where list of entities
    is a list of tuples like "(category name, text in tag)"
    """
    def get_entities_in_category_tags(self, texts):
        entities = []
        for text in texts:
            t_entities = re.findall(r'<category="(.+?)">(.+?)</category>', text)
            entities.append(t_entities)
        
        return entities

    """
    Get sequence of 0s and 1s based on the entities information of each text
    """
    def get_bin_dataset(self, dataset, entities):
        tokenized_dataset = self._nlp_tool.tokenize_texts(dataset, basic_tokenizer=True)
        tokenized_dataset = [[w[0] for w in text] for text in tokenized_dataset]
        
        bin_entities_per_text = []

        for text_i, tokens in enumerate(tokenized_dataset):
            len_tokens = len(tokens)

            bin_entities = np.zeros(len_tokens, dtype=int)

            token_i = 0
            for tup_entity in entities[text_i]:
                token_start = -1
                token_stop = -1
                
                entity = self._nlp_tool.tokenize_texts([tup_entity[1]], basic_tokenizer=True)[0]
                entity = [w[0] for w in entity]
                
                ent_i = 0
                while token_i < len_tokens:
                    ### Fixed values: there are some mismatches :/ ###
                    # TO DO: Fix tokenizer so I do not need to tokenize the entities separately
                    if tokens[token_i] == 'Iowa':
                        tokens[token_i] = 'Ia'
                    ###
                    
                    if entity[ent_i].lower() == tokens[token_i].lower():
                        if token_start == -1:
                            token_start = token_i
                        
                        ent_i = ent_i + 1
                        token_i = token_i + 1
                        
                        if ent_i == len(entity):
                            token_stop = token_i
                            break
                    else:
                        token_start = -1
                        token_i = token_i + 1
                
                if token_stop == -1:
                    raise Exception('Entity not found:', entity)
                else:
                    bin_entities[range(token_start, token_stop)] = 1

            bin_entities_per_text.append(bin_entities)
        
        return bin_entities_per_text
