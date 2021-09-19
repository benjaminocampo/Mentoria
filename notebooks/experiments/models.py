from embedding import create_embedding_matrix, create_embedding_layer
from encoding import create_vectorize_layer
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
import tensorflow as tf


def baseline_model(data, nof_classes, length_long_sentence, embedding_dim):
    vectorize_layer = create_vectorize_layer(length_long_sentence, "int")
    vectorize_layer.adapt(data.values)

    vocab = vectorize_layer.get_vocabulary()
    w2v_model = Word2Vec(sentences=data.apply(lambda s: s.split()),
                         min_count=10,
                         window=2,
                         vector_size=embedding_dim,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=5)

    embedding_matrix = create_embedding_matrix(w2v_model.wv, vocab, embedding_dim)

    embedding_layer = create_embedding_layer(vocab_size=len(vocab) + 1,
                                             embedding_dim=embedding_dim,
                                             embedding_matrix=embedding_matrix,
                                             input_length=length_long_sentence)

    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    # Agregar preprocesamiento

    model.add(vectorize_layer)
    model.add(embedding_layer)
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Flatten())
    model.add(Dense(units=nof_classes, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                     optimizer="adam",
                     metrics=["accuracy"])
    return model


def baseline_with_dropout_model(vectorize_layer, embedding_layer, nof_classes):
    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(embedding_layer)
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(rate=.4))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(rate=.4))
    model.add(Flatten())
    model.add(Dense(units=nof_classes, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                     optimizer="adam",
                     metrics=["accuracy"])
    return model


def baseline_with_batchnorm_model(vectorize_layer, embedding_layer,
                                  nof_classes):
    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(embedding_layer)
    model.add(Dense(units=256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(units=128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=nof_classes, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                     optimizer="adam",
                     metrics=["accuracy"])
    return model
