# %% [markdown]
# # Categorización de publicaciones de productos de Mercado Libre
#
# Autores: Maximiliano Tejerina, Eduardo Barseghian, Benjamín Ocampo 

# %% [markdown]
# ## Reducción del dataset y definición de funciones *helper*
# Luego de realizar la reducción del dataset a un total de 20 categorías y
# 646760 publicaciones, es colocado disponible en un servidor permitiendo ser
# accedido por medio de una URL.
#
# Inicialmente se definen constantes y funciones que permiten obtener
# información cuantitativa de las publicaciones así como de su estructura. Por
# un lado, `count_stopwords`, `count_special_chars`, `count_digits`, permiten
# hacer conteos sobre la estructura de las publicaciones, cuando en contraparte
# `proportion` permite hacer una comparación entre cantidades de grupos de
# interés. También se definen nuestras variables aleatorias o columnas
# relevantes y un conjunto `stopwords` que almacena palabras como articulos, o
# proposiciones frecuentes del español y el portugués.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords")


URL = "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"
df = pd.read_csv(URL)

title = "title"
category = "category"
label_quality = "label_quality"
language = "language"

stopwords = \
    set(nltk.corpus.stopwords.words("spanish")) | \
    set(nltk.corpus.stopwords.words("portuguese"))

def proportion(df, by, col):
    df_proportion = df.groupby([by, col]) \
        .agg(count=(col, "count")) \
        .join(df.groupby(by).size() \
        .to_frame()) \
        .rename(columns={0: "total"})

    df_proportion.loc[:, "proportion"] = df_proportion["count"] / df_proportion["total"]
    return df_proportion

count_stopwords = lambda s: sum(
    w.lower() in stopwords for w in word_tokenize(s)
)

def count_special_chars(s):
    word_freq = nltk.FreqDist(s)
    special_chars = "[@_!#$%^&*()<>?/\|}{~:]"
    return sum(word_freq[sc] for sc in special_chars)

def count_digits(s):
    word_freq =  nltk.FreqDist(s)
    digits = "0123456789"
    return sum(word_freq[d] for d in digits)

# %%
df.head(20)
# %% [markdown]
# ## Exploración de publicaciones
# Como primera iniciativa se optó por contabilizar la cantidad de publicaciones
# por categoría, con la finalidad de verificar si se persibía una diferencia de
# magnitud o variabilidad entre ellas.
# %%
fig = plt.figure(figsize=(12,8))
seaborn.countplot(data=df, x='category')
plt.xticks(rotation=80)
# %% [markdown]
# Notar que no es el caso para este dataset teniendo entre 30600 a 36000 de
# cantidad por cada categoría. Algo también a recalcar es la presencia de
# títulos repetidos para algunas de estas como se muestra en la siguiente tabla.
# %%
df[[category, title]].groupby(category).describe()
# %% [markdown]
# Si bien la frecuencia de estas repeticiones no es alta (de a lo sumo 2
# repeticiones), se dan en una pequeña fracción de títulos.
# %% [markdown]
# Otro dato de interés es la proporción de publicaciones en español y portugués
# que se obtiene a través de la función `proportion` que determina este
# resultado agrupando por categorías. No obstante, tampoco se obtuvo una
# gran diferencia.
# %%
df_language_by_category = proportion(df, category, language).reset_index()
fig = plt.figure(figsize=(15,7))
seaborn.barplot(
    y=df_language_by_category["count"],
    x=df_language_by_category["category"],
    hue=df_language_by_category["language"],
    ci=None
)
plt.xticks(rotation=70)
plt.ylabel("Cantidad de publicaciones")
plt.xlabel("Categorías de publicaciones")
plt.ticklabel_format(style='plain', axis='y')
# %%
df_language_by_category
# %% [markdown]
# También a partir de la siguiente tabla se puede ordenar observar la mínima y
# máxima proporción por categoría obteniendo que, de entre todas ellas, el
# idioma menos frecuente tiene al memos un 42% de las publicaciones. Las
# primeras 12 entre un 43% y un 57%. Y las restantes entre un 48% y un 53%.
# %%
df_language_by_category[[category, "proportion"]] \
    .groupby(category) \
    .agg(
        min_proportion=("proportion", "min"),
        max_proportion=("proportion", "max")
    ).sort_values(by="min_proportion")
# %% [markdown]
# Si se analiza para el total de publicaciones se obtiene que del total de
# 646760 publicaciones, 317768 (49.13%) son en español, 328992 (50.86%) son en
# portugués.
# %%
nof_items = len(df)
nof_items
# %%
nof_spanish_items = len(df[df[language] == "spanish"])
(nof_spanish_items, nof_spanish_items / nof_items * 100)
# %%
nof_portugues_items = nof_items - nof_spanish_items
(nof_portugues_items, nof_portugues_items / nof_items * 100)
# %% [markdown]
# ## Exploración de label quality
# Analogo al caso anterior se analiza la proporción de publicaciones `reliable`
# y `unreliable` agrupando por categoría. 
# %%
df_label_by_category = proportion(df, category, label_quality).reset_index()
fig = plt.figure(figsize=(8,6))
seaborn.barplot(
    y=df_label_by_category["count"],
    x=df_label_by_category["category"],
    hue=df_label_by_category["label_quality"],
    ci=None
)
plt.xticks(rotation=90)
plt.ticklabel_format(style='plain', axis='y')
# %%
df_label_by_category
# %% [markdown]
# ## Relación entre label quality e idioma
# %%
pd.crosstab(
    df[language],
    df[label_quality]
) / len(df)
# %% [markdown]
# ## Cantidad promedio de palabras del título por categoría

# %%
df_word_count = df[title].apply(lambda s: len(s.split(' '))) \
    .to_frame()\
    .rename(columns={"title": "word_count"})

df[[title, category]] \
    .join(df_word_count) \
    .groupby(category).agg(
        avg_word_count=("word_count", "mean")
)
# %%
df.join(df_word_count).describe()

# %% [markdown]
# ## Cantidad de stopwords por título

# %%
df[title] \
    .apply(count_stopwords) \
    .to_frame()\
    .rename(columns={"title": "nof_stopwords"}) \
    .join(df_word_count)
# %% [markdown]
# ## Cantidad de números por título
# %%
df[title] \
    .apply(count_digits) \
    .to_frame()\
    .rename(columns={"title": "nof_numbers"})
# %%
df[title] \
    .apply(count_special_chars) \
    .to_frame()\
    .rename(columns={"title": "nof_numbers"})

# %%
dff = df[title] \
    .agg([count_stopwords, count_digits, count_special_chars])