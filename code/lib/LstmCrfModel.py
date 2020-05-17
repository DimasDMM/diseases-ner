import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Reshape, Bidirectional, concatenate, Flatten
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss
from keras.utils import to_categorical

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from .ModelBase import ModelBase
from .NlpTool import NlpTool

class LstmCrfModel(ModelBase):
    WORD_TOKENIZER = 'word_tokenizer'
    POS_TOKENIZER = 'pos_tokenizer'
    
    def create_model(self, word_tokenizer, pos_tokenizer, sequence_length, model_params=None):
        if model_params is None:
            model_params = self._get_default_params()

        word_index = word_tokenizer.word_index
        pos_index = pos_tokenizer.word_index

        word_input = Input(shape=(sequence_length,), name='word_input')
        word_pipe = Embedding(input_dim=len(word_index) + 1,
                            output_dim=model_params['embedding_dim'],
                            input_length=sequence_length,
                            trainable=True)(word_input)
        word_pipe = Bidirectional(
                        LSTM(model_params['lstm_cells'],
                            return_sequences=True,
                            dropout=model_params['word_lstm_dropout'],
                            recurrent_dropout=model_params['word_lstm_rec_dropout']),
                        merge_mode='concat')(word_pipe)
        word_pipe = TimeDistributed(Flatten())(word_pipe)

        pos_input = Input(shape=(sequence_length,), name='pos_input')
        pos_pipe = Embedding(input_dim=len(pos_index) + 1,
                            output_dim=model_params['embedding_dim'],
                            input_length=sequence_length,
                            trainable=True)(pos_input)
        pos_pipe = Bidirectional(
                        LSTM(model_params['lstm_cells'],
                            return_sequences=True,
                            dropout=model_params['pos_lstm_dropout'],
                            recurrent_dropout=model_params['pos_lstm_rec_dropout']),
                        merge_mode='concat')(pos_pipe)
        pos_pipe = TimeDistributed(Flatten())(pos_pipe)
        
        # Concatenate both inputs
        comb_pipe = concatenate([word_pipe, pos_pipe])

        # Main BiLSTM model
        comb_pipe = Bidirectional(
            LSTM(model_params['lstm_cells'], return_sequences=True),
            merge_mode='concat')(comb_pipe)
        comb_pipe = TimeDistributed(Dense(64))(comb_pipe)
        
        output = CRF(2, name='output')(comb_pipe)
        
        model = Model(inputs=[word_input, pos_input], outputs=output)
        model.compile(
            loss=crf_loss,
            optimizer='adam',
            metrics=[crf_accuracy]
        )
        
        return model
    
    def load_model_weights(self, model_name, model):
        return self._file_manager.load_model_weights(model_name, model)
    
    def save_model_weights(self, model_name, model):
        self._file_manager.save_model_weights(model_name, model)
    
    def load_tokenizers(self, model_name):
        word_tokenizer = self._file_manager.load_tokenizer(model_name, self.WORD_TOKENIZER)
        pos_tokenizer = self._file_manager.load_tokenizer(model_name, self.POS_TOKENIZER)
        return word_tokenizer, pos_tokenizer
    
    def save_tokenizers(self, model_name, word_tokenizer, pos_tokenizer):
        self._file_manager.save_tokenizer(model_name, self.WORD_TOKENIZER, word_tokenizer)
        self._file_manager.save_tokenizer(model_name, self.POS_TOKENIZER, pos_tokenizer)
    
    def create_keras_tokenizers(self, dataset, num_words=150000):
        tok_dataset = [[w[0] for w in text] for text in dataset]
        pos_dataset = [[w[1] for w in text] for text in dataset]

        word_tokenizer = Tokenizer(num_words=num_words)
        word_tokenizer.fit_on_texts(tok_dataset)

        pos_tokenizer = Tokenizer(num_words=num_words)
        pos_tokenizer.fit_on_texts(pos_dataset)
        
        return word_tokenizer, pos_tokenizer
    
    def get_entities(self, text, model, word_tokenizer, pos_tokenizer):
        # Put the previous text inside of a list
        texts = list([text])

        # Tokenize it and get PoS tags
        nlp_text = self._nlp_tool.tokenize_texts(texts)[0]

        pred_bin_text = self._make_bin_pred(model, nlp_text, word_tokenizer, pos_tokenizer)
        pred_str_entities = self.bin_pred_to_str(nlp_text, pred_bin_text)
        
        return pred_str_entities

    def train_model(self, model, word_tokenizer, pos_tokenizer, train_dataset, bin_train_dataset, epochs=5):
        bin_train_dataset = self._data_to_sequences(bin_train_dataset, NlpTool.MAX_SEQUENCE_LENGTH)
        bincat_train_dataset = np.array([to_categorical(i, num_classes=2) for i in bin_train_dataset])
        
        tok_train_dataset = [[w[0] for w in text] for text in train_dataset]
        pos_train_dataset = [[w[1] for w in text] for text in train_dataset]
        
        tok_train_dataset = self._data_to_sequences(tok_train_dataset, NlpTool.MAX_SEQUENCE_LENGTH, word_tokenizer)
        pos_train_dataset = self._data_to_sequences(pos_train_dataset, NlpTool.MAX_SEQUENCE_LENGTH, pos_tokenizer)

        # Add early stop
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

        model.fit(
            {'word_input': tok_train_dataset, 'pos_input': pos_train_dataset},
            bincat_train_dataset,
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1
        )
        
        return model
    
    def _make_bin_pred(self, model, nlp_text, word_tokenizer, pos_tokenizer):
        tok_texts = list([[w[0] for w in nlp_text]])
        pos_texts = list([[w[1] for w in nlp_text]])

        tok_texts = self._data_to_sequences(tok_texts, NlpTool.MAX_SEQUENCE_LENGTH, word_tokenizer)
        pos_texts = self._data_to_sequences(pos_texts, NlpTool.MAX_SEQUENCE_LENGTH, pos_tokenizer)

        # Make the prediction
        pred_entities = model.predict({'word_input': tok_texts, 'pos_input': pos_texts})
        pred_entities = [np.argmax(s, axis=1) for s in pred_entities]
        
        return pred_entities[0]
    
    def _data_to_sequences(self, data, max_sequence_length, tokenizer=None):
        if tokenizer:
            data = tokenizer.texts_to_sequences(data)

        data = sequence.pad_sequences(data, maxlen=max_sequence_length, padding='post')
        return data
    
    def _get_default_params(self):
        model_params = {
            'embedding_dim': 100,
            'lstm_cells': 50,
            'word_lstm_dropout': 0.3,
            'word_lstm_rec_dropout': 0.3,
            'pos_lstm_dropout': 0.3,
            'pos_lstm_rec_dropout': 0.3
        }
        return model_params
