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
import re
from nltk import (download, corpus, tokenize)
from unidecode import unidecode
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import (text, sequence)

download("stopwords")
download('punkt')

stopwords = (set(corpus.stopwords.words("spanish")) |
             set(corpus.stopwords.words("portuguese")))


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

#
# def prepare_tokenizer(s):
#     word_tokenizer = Tokenizer()
#     word_tokenizer.fit_on_texts(s)
#     encoded_docs = t.texts_to_sequences()
#     return pad_sequences(encoded_docs)
#
# %%
URL = "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"
df = pd.read_csv(URL)
# %% [markdown]
"""
## Limpieza de Texto
"""
# %%
df = df.assign(cleaned_title=df["title"].apply(clean_text))
# %%
df[["title", "cleaned_title"]]
# %% [markdown]
"""
## Tokenización y Secuencias
TODO: Explicar que hace la función o como se realiza el encoding de los
títulos.
"""
# %%
word_tokenizer = text.Tokenizer()
word_tokenizer.fit_on_texts(df["cleaned_title"])
# %%
word_tokenizer.word_counts
# %%
word_tokenizer.word_index
# %%
nof_out_of_vocab_rep = 1
vocab_size = len(word_tokenizer.word_index) + nof_out_of_vocab_rep
# %%
encoded_sentences = word_tokenizer.texts_to_sequences(df["cleaned_title"])
# %%
encoded_sentences[:5]
# %%
encoded_sentences = sequence.pad_sequences(encoded_sentences,
                                                         padding='post')
# %%
encoded_sentences[:5]
# %% [markdown]
"""
## Label Encoding
"""
# %%
le = LabelEncoder()
encoded_labels = le.fit_transform(df["category"])
encoded_labels
# %%
le.classes_
# %% [markdown]
"""
## Word Embeddings
"""
# %%
