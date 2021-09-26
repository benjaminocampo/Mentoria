# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
# ---

# %% [markdown]
"""
# Categorización de publicaciones de productos de Mercado Libre

Autores: Maximiliano Tejerina, Eduardo Barseghian, Benjamín Ocampo

En `items_exploration.ipynb` se trabajó sobre un conjunto reducido de títulos
etiquetados de publicaciones de Mercado Libre, donde se analizó la
confiabilidad de las etiquetas, frecuencia de palabras, y dependencia de
variables, con el objetivo de obtener información que influya en la predicción
de la categoría más adecuada de un producto dado el título de su publicación.
Con el fin de utilizar modelos de *machine learning* en esta tarea, es
necesario realizar una codificación de los datos tabulados. Por ende, esta
notebook analiza los distintos pasos necesarios para obtener una matriz
resultante que pueda ser evaluada en modelos de clasificación.
"""
# %% [markdown]
"""
## Definición de funciones y constantes *helper*

Al igual que en la exploración, se definen algunas funcionalidades que serán de
utilidad durante la codificación, pero que no son muy relevantes para el
seguimiento de este trabajo. Con el fin de mantener la reproducibilidad con
herramientas *online* tales como `Google Colab`, son mantenidas en una sección
dentro de esta notebook. No obstante, en futuros trabajos se procederá a
estructurar dichas funcionalidades en scripts separados manteniendo un único
*entrypoint*.
"""
# %%
import re
import io
import tqdm
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Embedding, Dot
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from keras.models import Sequential
from keras.layers import Input

from sklearn.preprocessing import LabelEncoder
from unidecode import unidecode

from nltk import (download, corpus, tokenize)
from typing import List

download("stopwords")
download('punkt')
# %%
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


def clean_text(s: str, language: str) -> str:
    """
    Given a string @s and its @language the following parsing is performed:
        - Lowercase letters.
        - Removes non-ascii characters.
        - Removes numbers and special symbols.
        - Expand common contractions.
        - Removes stopwords.
    """
    numbers = r"\d+"
    symbols = r"[^a-zA-Z ]"
    # TODO: See if we can find more contractions.
    contractions = [
        ("c/u", "cada uno"),
        ("c/", "con"),
        ("p/", "para"),
    ]
    # TODO: Can we avoid chaining assignments? Maybe function composition.
    s = unidecode(s)
    s = s.lower()
    for expression, replacement in contractions:
        s = s.replace(expression, replacement)
    s = re.sub(numbers, "", s)
    s = re.sub(symbols, "", s)
    s = ' '.join(w for w in tokenize.word_tokenize(s)
                 if not w in corpus.stopwords.words(language))
    return s


def load_embedding(filename: str, vocab: List[str],
                   embedding_dim: int) -> Embedding:
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
        for line in f:
            word, *coef = line.rstrip().split(' ')
            embedding_indexes[word] = np.asarray(coef, dtype=float)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for index, word in enumerate(vocab):
        vector = embedding_indexes.get(word)
        if vector is not None:
            embedding_matrix[index] = vector
            nof_hits += 1
        else:
            nof_misses += 1

    embedding_layer = Embedding(
        vocab_size,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(
            embedding_matrix),
        trainable=False,
    )

    return embedding_layer, nof_hits, nof_misses

# %%
URL = "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"
df = pd.read_csv(URL)
# %% [markdown]
"""
## Limpieza de Texto

Si se observan algunos ejemplos de títulos en el conjunto de datos se puede ver
que varios de ellos utilizan números, símbolos especiales, o combinaciones de
mayúsculas y minúsculas con el fin de capturar la atención del lector. Sin
embargo, una palabra que es escrita de distintas maneras no es relevante para
catalogar dichas publicaciones. De manera similar ocurre con caracteres
especiales, números, e incluso artículos o proposiciones. Por ende, se procedió
a realizar un preprocesamiento de cada uno de los títulos donde se reemplazan:

 - Mayúsculas por minúsculas.
 - Caracteres que no tienen una codificación ascii.
 - Números y símbolos.
 - Contracciones de palabras por su expresión completa.
 - *Stopwords* de acuerdo al lenguage.

Esto es realizado por la función `clean_text` siendo aplicada para cada uno de
los títulos del conjunto de datos (Puede demorar algunos minutos).
"""
# %%
df["title"].sample(5)
# %%
df = df.sample(5000, random_state=0)
# %%
df = df.assign(cleaned_title=df[["title", "language"]].apply(
    lambda x: clean_text(*x), axis=1))
df[["title", "cleaned_title"]]
# %% [markdown]
"""
## Codificación de títulos: tokenización y secuencias

Una vez realizada la limpieza de los títulos, se procedió a identificar cada una
de las palabras que ocurren en ellos, confeccionando de esta manera un
vocabulario.
"""
# %%
word_tokenizer = text.Tokenizer()
word_tokenizer.fit_on_texts(df["cleaned_title"])
# %% [markdown]
"""
Esta identificación de palabras o *tokens* es realizada por una instancia de la
clase `Tokenizer` de `Keras` a partir de `fit_on_texts` dando lugar a algunas
propiedades y métodos, como la cantidad de oraciones utilizadas, frecuencia de
palabras, y orden por tokens más frecuentes.
"""
# %%
word_tokenizer.word_counts
# %%
word_tokenizer.word_index
# %% [markdown]
"""
En este caso, el atributo `word_index` muestra el vocabulario obtenido, donde a
cada palabra se le asigna un único índice según su frecuencia dada por
`word_counts`. Por ejemplo, el *token* `maquina` tiene mayor frecuencia que
todas las palabras en el vocabulario y por ende se le asigna el índice 1, luego
les siguen `x` con el índice 2, y así sucesivamente.
"""
# %%
encoded_titles = word_tokenizer.texts_to_sequences(df["cleaned_title"])
encoded_titles[:5]
# %% [markdown]
"""
Basándose en esto, se codificaron los títulos asignándole un vector
correspondiente a los índices de cada una de las palabras que lo componen por
medio de `texts_to_sequences`. Por ejemplo, el título `galoneira semi
industrial` se le asigna el vector `[580, 188, 40]`, ya que `galoneira`, `semi`,
e `industrial` son las palabras 580, 188, y 40 más frecuentes del vocabulario, es
decir, dichos números son los índices asignados a cada una de esas palabras como
puede a través de la siguiente celda.
"""
# %%
(
    word_tokenizer.word_index["galoneira"],
    word_tokenizer.word_index["semi"],
    word_tokenizer.word_index["industrial"]
)
# %% [markdown]
"""
Ahora bien, los vectores obtenidos tienen largos distintos según la cantidad de
palabras que un título presente. Dado que un modelo requiere datos de entrada de
una dimensión concreta, se rellenó con 0’s los vectores de representación hasta
obtener una longitud en común.
"""
# %%
encoded_titles = sequence.pad_sequences(encoded_titles, padding='post')
encoded_titles[:5]
# %%
encoded_titles.shape
# %% [markdown]
"""
## Codificación de títulos usando TextVectorizer

También se puede incluir la tokenización de palabras dentro del modelo por medio
una instancia de `TextVectorization`. Definiendo la cantidad de *tokens* o
palabras de nuestro vocabulario `max_tokens`, un tipo de codificación
`output_mode`, y el tamaño de la secuencia más larga, `output_sequence_length`,
se puede definir una capa `vectorize_layer` donde se realicen los pasos
descriptos en la subsección anterior como parte del modelo.
"""
# %%
length_long_sentence = (
    df["cleaned_title"]
        .apply(lambda s: s.split())
        .apply(lambda s: len(s)).max()
)
max_tokens = len(word_tokenizer.word_index)
vectorize_layer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=length_long_sentence)
# %% [markdown]
"""
De manera similar a `word_tokenizer`, debemos entrenar la capa pero esta vez
usando el método `adapt` con los títulos que son de interés.
"""
# %%
vectorize_layer.adapt(df["cleaned_title"].values)
encoded_titles = vectorize_layer(df["cleaned_title"])
encoded_titles[:5]
# %% [markdown]
"""
## Codificación de etiquetas: *Label encoding*

De igual manera que para los títulos, se necesitó codificar sus categorías
asignadas. Sin embargo, no fue por medio de secuencias de números sino más bien
a través de un índice por cada etiqueta, usando una instancia de la clase
`LabelEncoder` de la librería `scikit-learn`.
"""
# %%
le = LabelEncoder()
encoded_labels = le.fit_transform(df["category"])
encoded_labels
# %%
le.classes_
# %% [markdown]
"""
`encoded_labels` corresponde a la misma columna de datos `category` del
*dataframe* `df` con la diferencia de haber reemplazado cada etiqueta por un
número entre 0 y el total de categorías. Luego, en caso de querer realizar el
paso inverso de decodificación se puede usar el método `inverse_transorme` del
codificador `le`. Por ejemplo, la etiqueta cuyo índice es 15 corresponde con
`SEWING_MACHINES`.
"""
# %%
le.inverse_transform([15])
# %% [markdown]
"""
## *Word Embeddings*

En la sección anterior se representaron los títulos de las publicaciones por
medio de vectores densos, donde una palabra era representada por el índice $i$,
si esta era la $i$-ésima palabra más frecuente. No obstante, también se pueden
representar las palabras por medio de vectores que mantengan su semántica, es
decir, aquellas que tengan un significado similar, tendrán por consecuente una
representación vectorial similar.

Un *word embedding* es una matriz de parámetros entrenables donde a cada palabra
del vocabulario se le asigna un vector representante. En esta sección, se
procedió a obtener esta matriz con 2 métodos distintos:

 - Entrenando los parámetros por medio de los datos disponibles (*Custom*).
 - Utilizando representaciones disponibles en la web (*Pretrained*).
"""
# %% [markdown]
"""
### *Custom*

Para instanciar un *embedding* de las palabras del conjunto de datos, se
necesitó obtener la longitud del título más largo, el tamaño del vocabulario, y
la dimensión de los vectores representación.
"""
# %%
_, long_sentence_size = encoded_titles.shape
# %%
vocab_size = len(vectorize_layer.get_vocabulary())
output_dim = 64
# %% [markdown]
"""
Cabe recordar que el tamaño del vocabulario es la cantidad de palabras distintas
que ocurren en los títulos de las publicaciones más el número de dimensiones
para representar las palabras fuera del vocabulario.
"""
# %%
embedding_layer = Embedding(vocab_size,
                            output_dim,
                            input_length=long_sentence_size)
embedding_layer(tf.constant([1, 2]))
# %% [markdown]
"""
Una vez instanciado el *embedding* se puede evaluarlo en índices que representan
alguna palabra obteniendo la representación resultante. Por ejemplo, para los
números 1 y 2 se obtuvieron los vectores dados por la celda anterior.
"""
# %%
weights = embedding_layer.get_weights()[0]
vocab = word_tokenizer.word_index.keys()
# %% [markdown]
"""
Cabe recalcar nuevamente que los parámetros del *embedding* instanciado fueron
asignados de manera aleatoria y por lo tanto las representaciones de las
palabras no son acordes a su significado.
"""
# %% [markdown]
"""
### *Word2Vec*

Para entrenar los parámetros presentes en `weights` se utilizó el modelo
`word2vec` que consiste en generar *skip-grams* a través de una lista de
oraciones. Estos *skip-grams* son pares `(target, context)` donde dada una
palabra `target`, se le asocia otra denominada `context` que puede o no
encontrarse en su contexto. Si ocurre lo primero, se tiene un *skip-gram*
positivo, y en caso contrario uno negativo. El objetivo fue generar para cada
palabra en el vocabulario, un *skip-gram* positivo y una cantidad
`nof_negative_skipgrams` negativos, y utilizarlos para entrenar un *embedding*
personalizado. Varios aspectos de implementación fueron obtenidos desde las
siguientes fuentes:

 - [Artículo ilustrativo de
   word2vec](https://jalammar.github.io/illustrated-word2vec/)
 - [Tutorial de tensorflow y
   keras](https://www.tensorflow.org/tutorials/text/word2vec#generate_skip-grams_from_one_sentence)
"""
# %%
seed = 42
nof_negative_skipgrams = 4
batch_size = 1024
buffer_size = 10000
dataset = generate_skipgram_training_data(
    sequences=encoded_titles.numpy(),
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=seed,
    batch_size=batch_size,
    buffer_size=buffer_size)
# %% [markdown]
"""
Para la generación del conjunto de datos se utilizó la función
`generate_skipgram_training_data` con una cantidad de 4 *skip-grams* negativos
por 1 positivo y organizado en *mini-batches* para únicamente actualizar los
parámetros tras haber evaluado un lote o *batch* y agilizar el entrenamiento.

Debido al tamaño del conjunto de datos resultante, el entrenamiento puede llevar
algunos minutos. Por ello, se dispuso la matriz de pesos obtenida bajo la
semilla `seed` utilizada en un servidor para facilitar su acceso remoto.
También, pueden ser visualizados por la herramienta [Embedding
Projector](http://projector.tensorflow.org/). Para ello.

 - Seleccionar *Load data*.
 - Agregar los archivos
   [metadata.tsv](https://drive.google.com/file/d/1QkmlfADvogUfWcH2iPG7oZnjivu6_Hdd/view?usp=sharing)
   y
   [vectors.tsv](https://drive.google.com/file/d/1XW5ndaAocWhUFXZPoSEGLuxGjYbmLydW/view?usp=sharing).

En caso de no querer realizar el entrenamiento, se puede proceder a la siguiente
sección donde se utilizó el embedding pre-entrenado y otros disponibles online.
"""
# %%
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim, nof_negative_skipgrams)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
# %%
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = word_tokenizer.word_index.keys()
# %%
save_embedding(vocab, weights)
# %% [markdown]
"""
Luego de la etapa de entrenamiento. Se utilizó la función `save_embedding` para
almacenar en 2 archivos separados de manera estructurada para cada palabra en
`vocab` su representación vectorial.
"""
# %% [markdown]
"""
### *Pretrained*

Dado que en el conjunto de datos se encuentran títulos en español y portugués,
se utilizaron [word embeddings pre-entrenados con
FastText](https://fasttext.cc/docs/en/pretrained-vectors.html) de estos idiomas.

Para cargar el conjunto de datos se implementó la función `load_embedding` que
requiere el vocabulario a utilizar, la dimensión de los *embeddings* descargados
y la ruta del archivo local al ordenador en el que se está ejecutando esta
notebook. Por ejemplo, para cargar las representaciones de palabras en portugués
dado por el archivo `wiki.pt.vec` ubicado al mismo nivel de esta notebook
"""
# %%
filename = "wiki.pt.vec"
vocab = word_tokenizer.word_index.keys()
embedding_dim = 300
embedding_layer, nof_hits, nof_misses = load_embedding(filename, vocab, embedding_dim)
# %%
embedding_layer.embeddings_initializer.value
# %% [markdown]
"""
## Combinación de capas de preprocesamiento y embedding
Finalmente, se puede definir un modelo que realice primero la tokenización y luego su representación
usando las *embedding layers* *pretrained* o *custom*.
"""
# %%
embedding_layer = Embedding(len(vectorize_layer.get_vocabulary()),
                            embedding_dim,
                            weights=[weights],
                            trainable=False)

model = Sequential()
model.add(Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
model.add(embedding_layer)
# %%
input_data = df["cleaned_title"]
model.predict(input_data)
# %% [markdown]
"""
## Conclusión

La elección de un adecuado preprocesamiento y codificación pueden ser relevantes
durante la ejecución de un modelo. Debido a ello, se planteó una posible
representación pero sin dejar de lado otras posibilidades. En particular,
limpieza de texto sin eliminar *stopwords*, entrenamiento de *word embeddings*
utilizando *FastText*, o codificación de títulos por *one-hot encoding*. A su
vez, dado que se tienen títulos de dos idiomas distintos en el conjunto de
datos, si se utilizan *word embeddings* disponibles online, deben traerse de
estas dos fuentes y combinarse para formar una única matriz de vectores densos.
En este caso, la implementación actual solo permite extraer un *word embedding*
de una única fuente.

Otro de los aspectos que es importante mejorar es la disposición de los
*scripts* de `Python` y la jerarquía de directorios. Separarlos en directorios
`models` y `preprocess` separado del directorio `notebooks`, pero que tengan un
solo punto de acceso es otro de los trabajos a futuro.

Por último, resultaría interesante obtener una comparación entre *word
embeddings* ya pre-entrenados junto a otros personalizados de acuerdo al
conjunto de datos de Mercado Libre y si se observa alguna diferencia en los
modelos de categorización.
"""