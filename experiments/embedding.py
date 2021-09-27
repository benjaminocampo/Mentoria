from keras.layers import Embedding
from tqdm import tqdm
import numpy as np
from urllib.request import urlopen

def load_embedding(url: str) -> np.array:
    embedding_indexes = {}
    f = urlopen(url)
    _, _ = map(int, f.readline().split())
    for line in tqdm(f):
        line = line.decode("utf-8")
        word, *coef = line.rstrip().split(' ')
        embedding_indexes[word] = np.asarray(coef, dtype=float)

    return embedding_indexes

def create_embedding_matrix(model, vocab, embedding_dim):
    vocab_size = len(vocab) + 1
    nof_hits = 0
    nof_misses = 0

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for index, word in tqdm(enumerate(vocab)):
        if word in model:
            embedding_matrix[index] = model[word]
            nof_hits += 1
        else:
            nof_misses += 1
    return embedding_matrix

def create_embedding_layer(vocab_size, embedding_dim, embedding_matrix,
                           input_length):
    return Embedding(vocab_size,
                     embedding_dim,
                     weights=[embedding_matrix],
                     trainable=False,
                     input_length=input_length)