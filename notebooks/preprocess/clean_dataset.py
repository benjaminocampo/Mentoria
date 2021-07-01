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
"""
# %% [markdown]
"""
## Funciones y Constantes
"""
# %%
import pandas as pd
import numpy
import nltk
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# %%
nltk.download("stopwords")
nltk.download('punkt')

stopwords = \
    set(nltk.corpus.stopwords.words("spanish")) | \
    set(nltk.corpus.stopwords.words("portuguese"))


def remove_unimportant_words(s):
    """
    Removes from the string @s all the stopwords, digits, and special chars
    """
    special_chars = "-.+,[@_!#$%^&*()<>?/\|}{~:]"
    digits = "0123456789"
    invalid_chars = special_chars + digits

    reduced_title = ''.join(c for c in s if not c in invalid_chars)

    reduced_title = ' '.join(w.lower() for w in word_tokenize(reduced_title)
                             if not w.lower() in stopwords)
    return reduced_title


def expand_contractions(s):
    title = s
    contractions = {" c/u ": "cada uno", " p/": "para", " c/": "con"}
    for key, value in contractions.items():
        title = title.split(key)
        title = value.join(title)
    return title

def prepare_tokenizer(words):
    """
    funcion que genera un vocabulario, toma una lista de palabras.
    retorna una lista de palabras tokenizadas.
    """
    t = Tokenizer(filters='-.+,[@_!#$%^&*()<>?/\|}{~:]0123456789', lower=True)
    t.fit_on_texts(words)
    encoded_docs = t.texts_to_sequences(words)
    return pad_sequences(encoded_docs)
# %%
URL = "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"
df = pd.read_csv(URL)
# %% [markdown]
"""
## Limpieza de Texto
"""
# %% [markdown]
"""
## Tokenización y Secuencias
TODO: Explicar que hace la función o como se realiza el encoding de los
títulos.
"""
# %%
encoded_titles = prepare_tokenizer(df.title)
# %% [markdown]
"""
## Label Encoding
"""
# %%
le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(df["category"])
# %% [markdown]
"""
## Word Embeddings
"""
# %%
