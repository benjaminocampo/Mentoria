from embedding import create_embedding_matrix, create_embedding_layer
from encoding import create_vectorize_layer
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def ff_model(sentences, length_long_sentence, units, dropout, lr, embedding_dim):
    vectorize_layer = create_vectorize_layer(length_long_sentence, "int")
    vectorize_layer.adapt(sentences.to_numpy())
    vocab = vectorize_layer.get_vocabulary()

    # Embedding Layer
    w2v = Word2Vec(sentences=[s.split() for s in sentences.to_numpy()],
                   min_count=10,
                   window=2,
                   vector_size=embedding_dim,
                   alpha=0.03,
                   min_alpha=0.0007,
                   negative=5)

    embedding_matrix = create_embedding_matrix(w2v.wv, vocab, embedding_dim)

    embedding_layer = create_embedding_layer(vocab_size=len(vocab) + 1,
                                             embedding_dim=embedding_dim,
                                             embedding_matrix=embedding_matrix,
                                             input_length=length_long_sentence)

    # Model definition
    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(embedding_layer)
    model.add(Dense(units=units, activation="relu"))
    model.add(Dropout(rate=dropout))
    model.add(Flatten())
    model.add(Dense(units=20, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adam(learning_rate=lr),
                  metrics=["accuracy"])

    return model


def lstm_model(sentences, length_long_sentence, units, dropout, lr,
               embedding_dim):
    vectorize_layer = create_vectorize_layer(length_long_sentence, "int")
    vectorize_layer.adapt(sentences.to_numpy())
    vocab = vectorize_layer.get_vocabulary()

    # Embedding Layer
    w2v = Word2Vec(sentences=[s.split() for s in sentences.to_numpy()],
                   min_count=10,
                   window=2,
                   vector_size=embedding_dim,
                   alpha=0.03,
                   min_alpha=0.0007,
                   negative=5)

    embedding_matrix = create_embedding_matrix(w2v.wv, vocab, embedding_dim)

    embedding_layer = create_embedding_layer(vocab_size=len(vocab) + 1,
                                             embedding_dim=embedding_dim,
                                             embedding_matrix=embedding_matrix,
                                             input_length=length_long_sentence)


    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(units=units)))
    model.add(Dropout(rate=dropout))
    model.add(Flatten())
    model.add(Dense(units=20, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                     optimizer="adam",
                     metrics=["accuracy"])
    return model