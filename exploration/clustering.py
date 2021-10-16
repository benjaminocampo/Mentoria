# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
 # Categorización de publicaciones de productos de Mercado Libre

 Autores: Maximiliano Tejerina, Eduardo Barseghian, Benjamín Ocampo
"""
# %%
import pandas as pd
import fasttext
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

URL = "https://www.famaf.unc.edu.ar/~nocampo043/ML_2019_challenge_dataset_preprocessed.csv"
NOF_SAMPLES = 20000
SEED = 0

def unsupervised_data_gen(sentences, corpus_file):
    with open(corpus_file, "w") as out:
        for s in sentences:
            out.write(s + "\n")

def get_embedding(vocab, model):
    return (
        pd.DataFrame({word: model[word] for word in vocab})
        .transpose()
        .reset_index()
        .rename(columns={"index": "word"})
    )

def get_embedding_2DTSNE(vocab, model):
    embedding = get_embedding(vocab, model)
    X = embedding.drop(columns=["word"])
    X_TSNE = TSNE(n_components=2).fit_transform(X)
    embedding_TSNE = pd.concat(
        [pd.DataFrame(vocab, columns=["word"]),
         pd.DataFrame(X_TSNE)], axis=1)
    return embedding_TSNE

def plot_vocabulary_2DTSNE(X_TSNE,
                           vocab,
                           ax,
                           point_size=50,
                           legend_size=20,
                           tick_size=20,
                           annotation_size=20):
    nof_words = len(X_TSNE)
    ax.scatter(X_TSNE[:, 0],
               X_TSNE[:, 1],
               s=point_size)
    ax.set_title("t-SNE Plot", fontsize=legend_size)
    annot_list = []
    
    nof_words_to_annotate = 20
    for i in np.random.randint(nof_words, size=nof_words_to_annotate):
        a = ax.annotate(vocab[i], (X_TSNE[i, 0], X_TSNE[i, 1]),
                        size=annotation_size)
        annot_list.append(a)
    adjust_text(annot_list)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)


def plot_vocabulary_kmeans(X_TSNE,
                           km_model,
                           ax,
                           point_size=50,
                           marker_size=2,
                           legend_size=20,
                           tick_size=20):
    for k in range(km_model.n_clusters):
        ax.scatter(X_TSNE[km_model.labels_ == k, 0],
                   X_TSNE[km_model.labels_ == k, 1],
                   s=point_size,
                   label=k)
    ax.legend(bbox_to_anchor=(1.02, 1.02),
              loc='upper left',
              markerscale=marker_size,
              prop={"size": legend_size})
    ax.set_title("KMeans Plot", fontsize=legend_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

# %%
df = pd.read_csv(URL)
df = df.sample(n=NOF_SAMPLES, random_state=SEED)
df
# %%
sentences = df["cleaned_title"].tolist()
corpus_file = "titles.txt"

unsupervised_data_gen(sentences, corpus_file)
# %%
model = fasttext.train_unsupervised(corpus_file,
                                    model="cbow",
                                    lr=0.3,
                                    epoch=100,
                                    dim=100,
                                    wordNgrams=3,
                                    ws=3)

# %%
model.get_dimension()
# %%
model.get_words()[:30]
# %%
model.get_nearest_neighbors("cafetera")
# %% [markdown]
# ## Clustering a nivel de palabra
# %%
vocab = model.get_words()
embedding = get_embedding(vocab, model)
embedding_TSNE = get_embedding_2DTSNE(vocab, model)
X = embedding.drop(columns=["word"]).to_numpy()
X_TSNE = embedding_TSNE.drop(columns=["word"]).to_numpy()
# %%
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

_, (ax_kmeans, ax_tsne) = plt.subplots(1, 2, figsize=(30, 10))
plot_vocabulary_2DTSNE(X_TSNE, vocab, ax_tsne)
plot_vocabulary_kmeans(X_TSNE, kmeans, ax_kmeans)
ax_kmeans.grid()
ax_tsne.grid()
# %% [markdown]
# ## Clustering a nivel de título (oración)
# %%

