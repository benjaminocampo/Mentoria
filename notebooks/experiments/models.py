from embedding import create_embedding_matrix, create_embedding_layer
from encoding import create_vectorize_layer
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Bidirectional, LSTM, Flatten
from keras.optimizers import Adam
import tensorflow as tf



def build_model(hp, sentences, length_long_sentence):
    # Parameters
    hp_units_dense = hp.Int("units_dense", min_value=32, max_value=512, step=32)
    hp_dropout = hp.Float("dropout", 0.1, .6, sampling="log")
    # hp_units_LSTM = hp.Int("units_LSTM", min_value=32, max_value=512, step=32)
    hp_lr = hp.Float("lr", 1e-4, 1e-1, sampling="log")
    embedding_dim = 100

    # Vectorize Layer

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

    embedding_matrix = create_embedding_matrix(w2v.wv, vocab, 100)

    embedding_layer = create_embedding_layer(vocab_size=len(vocab) + 1,
                                             embedding_dim=embedding_dim,
                                             embedding_matrix=embedding_matrix,
                                             input_length=length_long_sentence)

    # Model definition
    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(embedding_layer)
    model.add(Dense(units=hp_units_dense, activation="relu"))
    model.add(Flatten())
    model.add(Dense(units=20, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=Adam(learning_rate=hp_lr),
                  metrics=["accuracy"])
    
    return model


def feed_forward_model(data, nof_classes, length_long_sentence, embedding_dim):
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
    model.add(vectorize_layer)
    model.add(embedding_layer)
    model.add(Dense(units=768, activation="relu", init="uniform"))
    model.add(Dropout(rate=.4))
    model.add(Dense(units=384, activation="relu", init="uniform"))
    model.add(Dropout(rate=.4))
    model.add(Dense(units=nof_classes, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                     optimizer="adam",
                     metrics=["accuracy"])
    return model


def LSTM_model(data, nof_classes, length_long_sentence,
                              embedding_dim):
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

    embedding_matrix = create_embedding_matrix(w2v_model.wv, vocab,
                                               embedding_dim)

    embedding_layer = create_embedding_layer(vocab_size=len(vocab) + 1,
                                             embedding_dim=embedding_dim,
                                             embedding_matrix=embedding_matrix,
                                             input_length=length_long_sentence)

    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorize_layer)
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(256)))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=nof_classes, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                     optimizer="adam",
                     metrics=["accuracy"])
    return model