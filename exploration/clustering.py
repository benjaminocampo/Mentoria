# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: 'Python 3.9.6 64-bit (''datasc'': conda)'
#     name: python3
# ---
# %% [markdown]
# # Categorización de publicaciones de productos de Mercado Libre
# Autores: Maximiliano Tejerina, Eduardo Barseghian, Benjamín Ocampo
# %% [markdown]
# Otra método de exploración de caracteristicas es mediante técnicas de
# aprendizaje no supervisado siendo útil para comprender aún mejor el conjunto
# de datos con el que se dispone y mejorar la etapa de clasificación con esta
# información.
#
# En particular, en esta notebook se trabajó sobre técnicas de *clustering* a
# partir de la codificación de los títulos obtenida por los embeddings.
# %% [markdown]
# ## Imports y funciones *helper* necesarias
# Nuevamente se hizo uso del conjunto de datos remoto que se trabajó en
# distintas secciones de este repositorio. En particular aquel que tuvo los
# títulos ya preprocesados. En caso de querer conocer más como se realizó esta
# etapa se explicaron a detalle en el directorio `encoding/`.
#
# A su vez, se utilizó una muestra de 20000 ejemplares del conjunto de datos
# para facilitar el desarrollo y la explicación de este trabajo.
# %%
import pandas as pd
import fasttext
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
# %%
URL = "https://www.famaf.unc.edu.ar/~nocampo043/ML_2019_challenge_dataset_preprocessed.csv"
NOF_SAMPLES = 20000
SEED = 0

def unsupervised_data_gen(sentences, corpus_file):
    with open(corpus_file, "w") as out:
        for s in sentences:
            out.write(s + "\n")

def get_word_embedding(vocab, vectorize):
    return (
        pd.DataFrame({word: vectorize(word) for word in vocab})
        .transpose()
        .reset_index()
        .rename(columns={"index": "word"})
    )

def get_word_embedding_2DTSNE(vocab, model):
    embedding = get_word_embedding(vocab, model)
    X = embedding.drop(columns=["word"])
    X_TSNE = TSNE(n_components=2).fit_transform(X)
    embedding_TSNE = pd.concat(
        [pd.DataFrame(vocab, columns=["word"]),
         pd.DataFrame(X_TSNE)], axis=1)
    return embedding_TSNE

def get_sentence_embedding(sentences, vectorize):
    return (
        pd.DataFrame({s: vectorize(s) for s in sentences})
        .transpose()
        .reset_index()
        .rename(columns={"index": "sentence"})
    )

def get_sentence_embedding_2DTSNE(sentences, vectorize):
    embedding = get_sentence_embedding(sentences, vectorize)
    X = embedding.drop(columns=["sentence"])
    X_TSNE = TSNE(n_components=2).fit_transform(X)
    embedding_TSNE = pd.concat(
        [pd.DataFrame(sentences, columns=["sentence"]),
         pd.DataFrame(X_TSNE)], axis=1)
    return embedding_TSNE


def plot_2DTSNE(X_TSNE,
                expressions,
                ax,
                point_size=50,
                legend_size=20,
                tick_size=20,
                annotation_size=20,
                annotate=True):
    nof_words = len(X_TSNE)
    ax.scatter(X_TSNE[:, 0],
               X_TSNE[:, 1],
               s=point_size)
    ax.set_title("t-SNE Plot", fontsize=legend_size)
    annot_list = []

    if annotate:
        nof_words_to_annotate = 20
        for i in np.random.randint(nof_words, size=nof_words_to_annotate):
            a = ax.annotate(expressions[i], (X_TSNE[i, 0], X_TSNE[i, 1]),
                            size=annotation_size)
            annot_list.append(a)
        adjust_text(annot_list)

    ax.tick_params(axis='both', which='major', labelsize=tick_size)


def plot_2D_kmeans(X_TSNE,
                   km_model,
                   ax,
                   marker_size=2,
                   legend_size=20,
                   tick_size=20):
    cluster_df = pd.DataFrame(X_TSNE)
    cluster_df = cluster_df.assign(label=km_model.labels_)
    sns.scatterplot(data=cluster_df, x=0, y=1, hue="label", palette="tab20", ax=ax)
    ax.legend(bbox_to_anchor=(1.02, 1.02),
              loc='upper left',
              markerscale=marker_size,
              prop={"size": legend_size})
    ax.set_title("KMeans Plot", fontsize=legend_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)


def plot_silhouette(X, km_model, ax, title_size=20, tick_size=20):
    visualizer = SilhouetteVisualizer(km_model, colors='tab20', ax=ax)
    visualizer.fit(X)
    ax.set_title("Silhoutte Plot", fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

def plot_elbow(X, estimator, metric, k_range, ax, title_size=10, tick_size=10):
    visualizer = KElbowVisualizer(estimator(),
                                  k=k_range,
                                  metric=metric,
                                  timings=False,
                                  ax=ax)
    visualizer.fit(X)
    ax.set_title("Elbow Plot", fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)


# %%
df = pd.read_csv(URL)
df = df.sample(n=NOF_SAMPLES, random_state=SEED)
df
# %% [markdown]
# Inicialmente se proyectó la columna `cleaned_title` con los títulos de las
# publicaciones ya preprocesadas siendo esta el corpus que se trabajó.
#
# Algo a recalcar es que luego del preprocesamiento se dió el caso de tener
# algunas oraciones repetidas. Para evitar tener dos representaciones distintas
# de la misma oración se optó por eliminar los duplicados.
# %%
sentences = df["cleaned_title"].drop_duplicates().tolist()
corpus_file = "titles.txt"
# %% [markdown]
# ## FastText
# A diferencia del trabajo efectuado de clasificación y aprendizaje supervisado,
# se utilizó FastText como algoritmo para obtener un modelo del lenguaje de los
# títulos por medio de la API de Facebook. Se pudo haber utilizado `Gensim` pero
# solo nos permite realizar codificación a nivel de palabra. Dado que el
# objetivo de este trabajo es aplicar técnicas de *clustering* sobre oraciones,
# la codificación de estas debía implementarse para esta librería. No fue este
# el caso durante el modelo usado en clasificación ya que Keras se encargó de
# obtener la representación vectorial de los títulos al incluir el modelo de
# lenguaje de `Gensim` como una *embedding layer*.
#
# FastText para obtener la representación a nivel de palabra genera una tarea de
# pretexto a partir de las oraciones que se le dispone por medio de un archivo,
# en este caso `corpus_file`. Con ello se obtiene el modelo de lenguaje que se
# utilizó para realizar esta exploración.
# %%
unsupervised_data_gen(sentences, corpus_file)
# %%
model = fasttext.train_unsupervised(corpus_file,
                                    model="cbow",
                                    lr=0.3,
                                    epoch=100,
                                    dim=100,
                                    wordNgrams=4,
                                    ws=4)
# %% [markdown]
# Cada palabra en el vocabulario tiene una representación vectorial de dimensión
# `dim=100`.
# %%
model.get_dimension()
# %%
model.get_words()[:30]
# %% [markdown]
# También como mostramos en otras notebooks de embeddings, la representación
# vectorial de las palabras `barbero` y `cafetera` está a una distancia cercana
# de otras con un significado similar.
# %%
model.get_nearest_neighbors("barbero")
# %%
model.get_nearest_neighbors("cafetera")
# %% [markdown]
# ## Clustering a nivel de palabra
# A pesar de este no ser el objetivo se intentó mostrar como se disponen las
# palabras en el espacio por medio de TSNE para proyectar los embeddings a 2D.
# Sin embargo, se aplicó `KMeans` en el espacio de 100 dimensiones.
# %%
vocab = model.get_words()
embedding = get_word_embedding(vocab, model.get_word_vector)
embedding_TSNE = get_word_embedding_2DTSNE(vocab, model.get_word_vector)
X = embedding.drop(columns=["word"]).to_numpy()
X_TSNE = embedding_TSNE.drop(columns=["word"]).to_numpy()
# %%
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
# %%
_, (ax_kmeans, ax_tsne) = plt.subplots(1, 2, figsize=(30, 10))
plot_2DTSNE(X_TSNE, vocab, ax_tsne)
plot_2D_kmeans(X_TSNE, kmeans, ax_kmeans)
ax_kmeans.grid()
ax_tsne.grid()
plt.show()
# %%
_, (ax_silhouette, ax_kmeans) = plt.subplots(1, 2, figsize=(30, 10))
plot_silhouette(X, kmeans, ax_silhouette)
plot_2D_kmeans(X_TSNE, kmeans, ax_kmeans)
ax_silhouette.grid()
ax_kmeans.grid()
plt.show()
# %% [markdown]
# Sin embargo a pesar de los gráficos, no se alcanza a percibir agrupamientos a
# través de la representación con TSNE. Sin embargo, la cercania de los
# significados se mantiene. Se cree que la disposición de las palabras en el
# `blob` si ocurren. Por ejemplo que las palabras más relacionadas con
# `cafetera` se encuentran en el hemisferio izquierdo del `blob`.
# %% [markdown]
# ## Clustering a nivel de títulos
# Para esta sección se obtuvo el embedding de una oración tomando las
# representaciones individuales de cada palabra que la componen, dividiendo esos
# vectores por su norma y considerando el promedio de ellas. Esto realiza
# internamente FastText por medio de `get_sentence_vector`. 
# %%
embedding = get_sentence_embedding(sentences, model.get_sentence_vector)
embedding_TSNE = get_sentence_embedding_2DTSNE(sentences,
                                               model.get_sentence_vector)
X = embedding.drop(columns=["sentence"]).to_numpy()
X_TSNE = embedding_TSNE.drop(columns=["sentence"]).to_numpy()
# %%
fig, ax_elbow = plt.subplots()
k_range = (2, 20)
plot_elbow(X, KMeans, "distortion", k_range, ax_elbow)
plt.show()
# %%
kmeans = KMeans(n_clusters=11)
kmeans.fit(X)
# %%
_, (ax_silhouette, ax_kmeans) = plt.subplots(1, 2, figsize=(30, 10))
plot_silhouette(X, kmeans, ax_silhouette)
plot_2D_kmeans(X_TSNE, kmeans, ax_kmeans)
ax_silhouette.grid()
ax_kmeans.grid()
plt.show()