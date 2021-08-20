import numpy as np
from typing import List
from tqdm import tqdm

def load_embedding(filename: str, vocab: List[str],
                   embedding_dim: int) -> np.array:
    """
    Given a path in your local system of an embedding file where each line has
    the embedded vector of dimension @embedding_dim separated by spaces, returns
    an embedding that contains only the words in @vocab.
    """
    vocab_size = len(vocab) + 1
    nof_hits = 0
    nof_misses = 0

    embedding_indexes = {}
    with open(filename) as f:
        _, _ = map(int, f.readline().split())
        for line in tqdm(f):
            word, *coef = line.rstrip().split(' ')
            embedding_indexes[word] = np.asarray(coef, dtype=float)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for index, word in tqdm(enumerate(vocab)):
        vector = embedding_indexes.get(word)
        if vector is not None:
            embedding_matrix[index] = vector
            nof_hits += 1
        else:
            nof_misses += 1

    return embedding_matrix, nof_hits, nof_misses