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
seguimiento de este trabajo. No obstante, con el fin de mantener la
reproducibilidad con herramientas *online* tales como `Google Colab`, son
mantenidas en una sección dentro de esta notebook.
"""
# %%
import pandas as pd
import re
from keras.layers.embeddings import Embedding
from keras.preprocessing import (text, sequence)
from nltk import (download, corpus, tokenize)
from sklearn.preprocessing import LabelEncoder
from unidecode import unidecode

download("stopwords")
download('punkt')

stopwords = (set(corpus.stopwords.words("spanish")) |
             set(corpus.stopwords.words("portuguese")))

URL = "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"
df = pd.read_csv(URL)

def clean_text(s: str) -> str:
    """
    Given a string @s the following parsing is performed:
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
    s = ' '.join(w for w in tokenize.word_tokenize(s) if not w in stopwords)
    return s
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
 - *Stopwords*

Esto es realizado por la función `clean_text` siendo aplicada para cada uno de
los títulos del conjunto de datos.
"""
# %%
df["title"].sample(5)
# %%
df = df.assign(cleaned_title=df["title"].apply(clean_text))
# %%
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
industrial` se le asigna el vector `[576, 186, 40]`, ya que `galoneira`, `semi`,
e `industrial` son las palabras 576, 186, y 40 más frecuente del vocabulario, es
decir, `word_index[galoneira] = 576`, `word_index[semi] = 186`, y
`word_index[industrial] = 40`.

Ahora bien, los vectores obtenidos tienen largos distintos según la cantidad de
palabras que un título presente. Dado que un modelo requiere datos de entrada de
una dimensión concreta, se rellenó con 0’s los vectores de representación de las
palabras hasta obtener una longitud en común.
"""
# %%
encoded_titles = sequence.pad_sequences(encoded_titles, padding='post')
encoded_titles[:5]
# %%
encoded_titles.shape
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
"""
# %% [markdown]
"""
### Custom
"""
# %%
# %%
nof_out_of_vocab_rep = 1
vocab_size = len(word_tokenizer.word_index) + nof_out_of_vocab_rep
_, length_long_sentence = encoded_titles.shape
output_dim = 64
embedding = Embedding(vocab_size, output_dim, input_length=length_long_sentence)
# %% [markdown]
"""
### Pretrained
"""
# %%
