from loguru import logger

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

class SpamEmailDetect():
    def __init__(self):
        bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
        bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

        text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'Inputs')
        preprocessed_text = bert_preprocessor(text_input)
        embeded = bert_encoder(preprocessed_text)
        dropout = tf.keras.layers.Dropout(0.1, name = 'Dropout')(embeded['pooled_output'])
        outputs = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'Dense')(dropout)

        self.model = tf.keras.Model(inputs = [text_input], outputs = [outputs])

        self.metrics = [
            tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
            tf.keras.metrics.Precision(name = 'precision'),
            tf.keras.metrics.Recall(name = 'recall')
        ]
    
    def summary(self):
        '''
        Print a useful summary of model.
        '''
        return self.model.summary()
    
    def compile(self):
        '''
        Compile model
        '''
        self.model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
                metrics = self.metrics
        )
    
    def fit(self, X_train, y_train, epochs: int=10):
        '''
        Fit data into model
        '''
        self.model.fit(X_train, y_train, epochs=epochs)
    
    def evaluate(self, X_test, y_test):
        '''
        Evaluate the model
        '''
        self.model.evaluate(X_test,  y_test)
    
    def predict(self, X_test):
        '''
        Predict results given on the input.
        '''
        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred
        return y_pred