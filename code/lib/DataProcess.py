import re
import numpy as np
import spacy
from spacy.symbols import ORTH, LEMMA, POS
from collections import Counter
from keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence

class DataProcess:
    MAX_SEQUENCE_LENGTH = 100
    ENTITY_START = '<entity>'
    ENTITY_END = '</entity>'
    UNK = '<UNK>'
    
    def __init__(self, jaccard):
        self._jaccard = jaccard
    
    def apply_initial_cleaner(self, texts):
        texts = [re.sub(r'^[0-9]+\s*', '', text) for text in texts]
        texts = [re.sub(r'<category=".+?">(.+?)</category>', r'{}\1{}'.format(self.ENTITY_START, self.ENTITY_END), text) for text in texts]
        return texts

    def get_indicator_sequences(self, tok_dataset, skip_entity_tags=True):
        indicators = []
        
        for text in tok_dataset:
            t_indicator = []
            
            ind_enabled = False
            for token in text:
                ind_enabled = token[2].text == self.ENTITY_START or ind_enabled and token[2].text != self.ENTITY_END
                
                if skip_entity_tags and token[2].text in [self.ENTITY_START, self.ENTITY_END]:
                    continue
                
                t_indicator.append(int(ind_enabled))
            
            indicators.append(np.array(t_indicator))
        
        return np.array(indicators)

    """
    Returns the words of the entities
    """
    def get_entities(self, tok_dataset):
        entities = []
        
        for text in tok_dataset:
            entity_start = None
            t_entities = []
            
            for i, t in enumerate(text):
                if t[2].text == self.ENTITY_START:
                    entity_start = i
                elif t[2].text == self.ENTITY_END:
                    entity_end = i
                    try:
                        entity = text[(entity_start + 1):i]
                    except Exception as e:
                        print(i)
                        print(t)
                        raise e
                    t_entities.append([e[0] for e in entity])
                    entity_start = None
            
            entities.append(t_entities)
        
        return entities

    def find_entities(self, tok_dataset, tok_entities, min_score=0.5):
        entities_per_text = []
        indicators_per_text = []
        
        for text in tok_dataset:
            text_len = len(text)
            t_entities = []
            t_indicator = []
            
            i = 0
            while i < text_len:
                entity_found = False
                
                for entity in tok_entities:
                    entity_len = len(entity)
                    score = 0
                    
                    if entity_len + i > text_len:
                        # The entity cannot fit in the tokenized words
                        continue

                    k = 0
                    while k < entity_len:
                        score = score + self._jaccard.word_jaccard(entity[k][2].text, text[i + k][2].text)
                        k = k + 1
                    
                    score = score / entity_len
                    if score >= min_score:
                        entity_found = True
                        t_entities.append(text[i:i+k])
                        t_indicator = t_indicator + [1]*k
                        i = i + k - 1
                
                if not entity_found:
                    t_indicator.append(0)
                
                i = i + 1
            
            entities_per_text.append(t_entities)
            indicators_per_text.append(t_indicator)
        
        return np.array(entities_per_text), np.array(indicators_per_text)

    def get_texts_words(self, tok_dataset, skip_entity_tags=True):
        if skip_entity_tags:
            skip_tokens = [self.ENTITY_START, self.ENTITY_END]
        else:
            skip_tokens = []
        
        texts_words = [[token[0] for token in text if token[0] not in skip_tokens] for text in tok_dataset]
        return np.array(texts_words)

    def get_texts_pos(self, tok_dataset, skip_entity_tags=True):
        if skip_entity_tags:
            skip_tokens = [self.ENTITY_START, self.ENTITY_END]
        else:
            skip_tokens = []
        
        texts_words = [[token[1] for token in text if token[1] not in skip_tokens] for text in tok_dataset]
        return np.array(texts_words)
    
    def encode_texts(self, dataset_texts, word2id):
        encoded_data_words = [[word2id[w] if w in word2id else word2id[self.UNK] for w in text] for text in dataset_texts]    
        return np.array(encoded_data_words)
    
    def get_nlp_engine(self):
        nlp = spacy.load('en_core_web_sm')
        
        nlp.tokenizer.add_special_case(self.ENTITY_START, [{ORTH: self.ENTITY_START, LEMMA: u'<entity>', POS: u'X'}])
        nlp.tokenizer.add_special_case(self.ENTITY_END, [{ORTH: self.ENTITY_END, LEMMA: u'</entity>', POS: u'X'}])
        
        return nlp

    def get_tokens(self, nlp_tokens):
        pos = [w.pos_ for w in nlp_tokens]
        words = [w.lemma_.strip() if w.lemma_ != "-PRON-" else w.lower_.strip() for w in nlp_tokens]
        tokens = [(words[i].lower(), pos[i], nlp_tokens[i]) for i in range(len(nlp_tokens))]
            
        # Remove empty tokens
        tokens = [token for token in tokens if token[0].strip() != '']

        return tokens

    def tokenize_texts(self, dataset, split_sentences=True):
        tok_dataset = []
        
        nlp = self.get_nlp_engine()
        e_start = self.ENTITY_START
        e_end = self.ENTITY_END
        
        for text in dataset:
            text = re.sub(rf"({e_start}|{e_end})", r' \1 ', text, re.UNICODE)
            text = re.sub(r"\s+", ' ', text.strip())  # Normalize white spaces
            nlp_tokens = nlp(text)
            
            if split_sentences:
                sentences = []
                
                entity_indicator = 0
                sentence_stack = []
                
                for s in nlp_tokens.sents:
                    for token in s:
                        if token.text == self.ENTITY_START:
                            entity_indicator = entity_indicator + 1
                        elif token.text == self.ENTITY_END:
                            entity_indicator = entity_indicator - 1
                    
                    sentence_stack = sentence_stack + list(s)
                    if entity_indicator == 0:
                        sentences.append(sentence_stack)
                        sentence_stack = []
                
                for s in sentences:
                    tokens = self.get_tokens(s)
                    tok_dataset.append(tokens)
            else:
                tokens = self.get_tokens(nlp_tokens)
                tok_dataset.append(tokens)
        
        return np.array(tok_dataset)

    def get_vocab_dictionary(self, data_words, additional_words=[], num_words=150000, min_count=1):
        counter_words = Counter()
        for words in data_words:
            counter_words.update(words)

        vocab_words_count = {w for w, c in counter_words.items() if c >= min_count}
        vocab_words = sorted(list(vocab_words_count))
        
        vocab_words = [''] + list(additional_words) + vocab_words
        if len(vocab_words) - 1 > num_words:
            vocab_words = vocab_words[1:(num_words + 1)]
        
        word2id = {word.strip(): idx for idx, word in enumerate(vocab_words)}
        
        del word2id['']
        del vocab_words[0]
        
        vocab_size = len(vocab_words)
        
        return vocab_words, word2id, vocab_size

    def to_categorical(self, binary_data, num_classes=2):
        return np.array([to_categorical(i, num_classes=num_classes) for i in binary_data])

    def to_sequences(self, data, max_sequence_length, tokenizer=None):
        data = sequence.pad_sequences(data, maxlen=max_sequence_length, padding='post')
        return np.array(data)

    def bin_to_str(self, tok_dataset, indicators):
        entities = []
        
        for text_i, text in enumerate(tok_dataset):
            t_entities = []
            
            stack = []
            for token_i, token in enumerate(text):
                if indicators[text_i][token_i]:
                    stack.append(token[2].text_with_ws)
                elif len(stack) > 0:
                    t_entities.append(''.join(stack).strip())
                    stack = []
            
            if len(stack) > 0:
                t_entities.append(''.join(stack).strip())
            
            entities.append(t_entities)
        
        return entities
