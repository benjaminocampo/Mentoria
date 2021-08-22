import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Embedding, Dot
from tensorflow.keras.preprocessing import sequence

import io
import tqdm
from typing import List

import pandas as pd
import numpy as np

class Word2Vec(Model):
    """
    word2vec class that implements @call method that will be called during
    training.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, num_ns: int):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(vocab_size,
                                          embedding_dim,
                                          input_length=1,
                                          name="w2v_embedding")
        self.context_embedding = Embedding(vocab_size,
                                           embedding_dim,
                                           input_length=num_ns + 1)
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = self.dots([context_emb, word_emb])
        return self.flatten(dots)

    
def generate_skipgram_training_data(sequences: np.array, window_size: int,
                                    num_ns: int, vocab_size: int,
                                    batch_size: int, buffer_size: int,
                                    seed: int) -> tf.Tensor:
    """
    Generates skip-gram pairs with negative sampling for a list of sequences
    (int-encoded sentences) based on window size, number of negative samples
    and vocabulary size.
    """
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
        vocab_size)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm.tqdm(sequences):

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates],
                                0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size).batch(batch_size,
                                                 drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def save_embedding(vocab: List[str], weights: np.array) -> None:
    """
    Given a list of words and a its corresponding weights, it saves them in
    files metadata.tsv and vectors.tsv respectively.
    """
    out_vectors = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_metadata = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index != 0:
            vec = weights[index]
            out_vectors.write('\t'.join([str(x) for x in vec]) + "\n")
            out_metadata.write(word + "\n")
    out_vectors.close()
    out_metadata.close()   


def customised_embedding(encoded_titles,vocab):
    _, long_sentence_size = y_train.shape
    nof_dim_out_of_vocab = 1
    vocab_size = len(vocab) + nof_dim_out_of_vocab
    output_dim = 64

    embedding_layer = Embedding(vocab_size, output_dim, input_length=long_sentence_size)

    weights = embedding_layer.get_weights()[0]

    """
        Para entrenar los parámetros presentes en `weights` se usó `word2vec`, que genera *skip-grams* mediante una lista de oraciones. 

        Cada *skip-grams* es un par de palabras `(target, context)` donde a cada palabra `target`, se le asocia otra denominada `context` 
        que puede o no encontrarse en su contexto. Así se tienen *skip-grams* positivo y negativos. 

        Para cada palabra en el vocabulario, se genera un *skip-gram* positivo y una cantidad
        `nof_negative_skipgrams` negativos. Se los usa para entrenar un *embedding* personalizado. 
    """

    seed = 42
    nof_negative_skipgrams = 4
    
    dataset = generate_skipgram_training_data(
                sequences=encoded_titles,
                window_size=2,
                num_ns=4,
                vocab_size=vocab_size,
                seed=seed,
                batch_size=1024,
                buffer_size=10000
                )

    embedding_dim = 128
    word2vec = Word2Vec(vocab_size, embedding_dim, nof_negative_skipgrams)
    word2vec.compile(optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy']
                    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])

    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    
    return weights 
    
    #save_embedding(vocab, weights)
    #Se utilizó la función `save_embedding` para almacenar en 2 archivos separados de manera estructurada para cada palabra en vocab su representación vectorial.