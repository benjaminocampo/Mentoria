from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM
from sklearn.base import BaseEstimator
import numpy as np
import tensorflow as tf


class LSTM_Net(BaseEstimator):
    def __init__(self,
                 vector_dim,
                 sequence_dim,
                 units=256,
                 dropout=0,
                 lr=1e-3,
                 batch_size=1000,
                 epochs=10):
        self.units = units
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_dim = sequence_dim
        self.vector_dim = vector_dim
        self.model = Sequential()
        self.model.add(Input(shape=(sequence_dim, vector_dim), dtype=float))
        self.model.add(SpatialDropout1D(rate=dropout))
        self.model.add(LSTM(units=units))
        self.model.add(Dense(units=20, activation="softmax"))
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer=Adam(learning_rate=lr),
                           metrics=["accuracy"])

    def fit(self, X, y):
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class FeedForward_Net(BaseEstimator):
    def __init__(self,
                 vector_dim,
                 units=256,
                 dropout=0,
                 lr=1e-3,
                 batch_size=1000,
                 epochs=10):
        self.units = units
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.vector_dim = vector_dim
        self.model = Sequential()
        self.model.add(Input(shape=(vector_dim, ), dtype=float))
        self.model.add(Dense(units=units, activation="relu"))
        self.model.add(Dropout(rate=dropout))
        self.model.add(Dense(units=20, activation="softmax"))
        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer=Adam(learning_rate=lr),
                           metrics=["accuracy"])

    def fit(self, X, y=None):
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
