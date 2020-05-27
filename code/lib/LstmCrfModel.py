import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed, Reshape, Bidirectional, concatenate, Flatten
from keras_contrib.layers import CRF
from keras_contrib.metrics import crf_accuracy
from keras_contrib.losses import crf_loss

from .ModelBase import ModelBase

class LstmCrfModel(ModelBase):
    WORD_VOCAB = 'word_tokenizer'
    POS_VOCAB = 'pos_tokenizer'
    
    def create_model(self, vocab_words_size, vocab_pos_size, sequence_length, model_params=None):
        if model_params is None:
            model_params = self._get_default_params()
        
        word_input = Input(shape=(sequence_length,), name='words_input')
        word_pipe = Embedding(input_dim=vocab_words_size + 1,
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
        pos_pipe = Embedding(input_dim=vocab_pos_size + 1,
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
        self._file_manager.load_model_weights(model_name, model)
    
    def save_model_weights(self, model_name, model):
        self._file_manager.save_model_weights(model_name, model)
    
    def load_vocabularies(self, model_name):
        word_vocab = self._file_manager.load_vocabulary(model_name, self.WORD_VOCAB)
        pos_vocab = self._file_manager.load_vocabulary(model_name, self.POS_VOCAB)
        
        word2id = {word:id for id, word in enumerate(word_vocab)}
        pos2id = {pos:id for id, pos in enumerate(pos_vocab)}
        
        vocab_words_size = len(word_vocab)
        vocab_pos_size = len(pos_vocab)
        
        return (word_vocab, word2id, vocab_words_size), (pos_vocab, pos2id, vocab_pos_size)
    
    def save_vocabularies(self, model_name, word_vocab, pos_vocab):
        self._file_manager.save_vocabulary(model_name, self.WORD_VOCAB, word_vocab)
        self._file_manager.save_vocabulary(model_name, self.POS_VOCAB, pos_vocab)
    
    def make_bin_prediction(self, model, words_enc, pos_enc):
        pred_indicators_cat = model.predict({'words_input': words_enc, 'pos_input': pos_enc})
        pred_indicators = np.array([np.argmax(s, axis=-1) for s in pred_indicators_cat])
        return pred_indicators

    def train_model(self, model, vocab_words, vocab_pos, train_words_enc, train_pos_enc, train_indicators_cat, epochs=5):
        # Add early stop
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

        model.fit(
            {'words_input': train_words_enc, 'pos_input': train_pos_enc},
            train_indicators_cat,
            epochs=epochs,
            callbacks=[early_stop],
            verbose=1
        )
        
        return model
    
    def _get_default_params(self):
        model_params = {
            'embedding_dim': 100,
            'lstm_cells': 128,
            'word_lstm_dropout': 0.3,
            'word_lstm_rec_dropout': 0.3,
            'pos_lstm_dropout': 0.3,
            'pos_lstm_rec_dropout': 0.3
        }
        return model_params
