# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
#  # Categorización de publicaciones de productos de Mercado Libre
#
#  Autores: Maximiliano Tejerina, Eduardo Barseghian, Benjamín Ocampo

# %% [markdown]
#  ## Reducción del dataset y definición de funciones *helper*
#  Luego de realizar la reducción del dataset a un total de 20 categorías y
#  646760 publicaciones, es colocado disponible en un servidor permitiendo ser
#  accedido por medio de una URL.
#
#  Inicialmente se definen constantes y funciones que permiten obtener
#  información cuantitativa de las publicaciones así como de su estructura. Por
#  un lado, `count_stopwords`, `count_special_chars`, `count_digits`, permiten
#  hacer conteos sobre la estructura de las publicaciones, cuando en contraparte
#  `proportion` permite hacer una comparación entre cantidades de grupos de
#  interés. También se definen nuestras variables aleatorias o columnas
#  relevantes y un conjunto `stopwords` que almacena palabras como articulos, o
#  proposiciones frecuentes del español y el portugués.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("stopwords")
nltk.download('punkt')


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
    """Returns a dataframe that has:
        1 The number of rows given by the group formed by @col and @by.
        2 The number of rows given by the group @by.
        3 The proportion of 1 related to 2.
    """
    df_proportion = df.groupby([by, col]) \
        .agg(count=(col, "count")) \
        .join(df.groupby(by).size().to_frame()) \
        .rename(columns={0: "total"})

    df_proportion.loc[:, "proportion"] = df_proportion["count"] / df_proportion["total"]
    return df_proportion

def count_words(s):
    """
    Counts the number of words in the string @s
    """
    return len(s.split())

def count_stopwords(s):
    """
    Counts the number of stopwords in the string @s
    """
    return sum(
        w.lower() in stopwords for w in word_tokenize(s)
    )

def count_special_chars(s):
    """
    Counts the number of special chars in the string @s
    """
    word_freq = nltk.FreqDist(s)
    special_chars = "-.+,[@_!#$%^&*()<>?/\|}{~:]"
    return sum(word_freq[sc] for sc in special_chars)

def count_digits(s):
    """
    Counts the number of digits in the string @s
    """
    word_freq =  nltk.FreqDist(s)
    digits = "0123456789"
    return sum(word_freq[d] for d in digits)

def remove_unimportant_words(s):
    """
    Removes from the string @s all the stopwords, digits, and special chars
    """
    special_chars = "-.+,[@_!#$%^&*()<>?/\|}{~:]"
    digits = "0123456789"
    invalid_chars = special_chars + digits

    reduced_title = ''.join(c for c in s if not c in invalid_chars)

    reduced_title = ' '.join(
        w.lower() for w in word_tokenize(reduced_title)
        if not w.lower() in stopwords
    )
    return reduced_title


# %%
df.head(20)

# %% [markdown]
#  ## Exploración de publicaciones
#  Como primera iniciativa se optó por contabilizar la cantidad de publicaciones
#  por categoría, con la finalidad de verificar si se persibía una diferencia de
#  magnitud o variabilidad entre ellas.

# %%
fig = plt.figure(figsize=(12,8))
seaborn.countplot(data=df, x='category', color="salmon")
plt.xticks(rotation=80)

# %% [markdown]
#  Notar que no es el caso para este dataset teniendo entre 30600 a 36000 de
#  cantidad por cada categoría. Algo también a recalcar es la presencia de
#  títulos repetidos para algunas de estas como se muestra en la siguiente tabla.

# %%
df[[category, title]].groupby(category).describe()

# %% [markdown]
#  Si bien la frecuencia de estas repeticiones no es alta (de a lo sumo 2
#  repeticiones), se dan en una pequeña fracción de títulos.

# %% [markdown]
#  Otro dato de interés es la proporción de publicaciones en español y portugués
#  que se obtiene a través de la función `proportion` que determina este
#  resultado agrupando por categorías. No obstante, tampoco se obtuvo una
#  gran diferencia.

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
#  También a partir de la siguiente tabla se puede observar la mínima y máxima
#  proporción por categoría obteniendo que, de entre todas ellas, el idioma menos
#  frecuente tiene al menos un 42% de las publicaciones. Las primeras 12 entre un
#  43% y un 57%. Y las restantes entre un 48% y un 53%.

# %%
df_language_by_category[[category, "proportion"]] \
    .groupby(category) \
    .agg(
        min_proportion=("proportion", "min"),
        max_proportion=("proportion", "max")
    ).sort_values(by="min_proportion")

# %% [markdown]
#  Si se analiza para el total de publicaciones se obtiene que del total de
#  646760 publicaciones, 317768 (49.13%) son en español, 328992 (50.86%) son en
#  portugués.

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
#  ## Exploración de label quality
#  Analogo al caso anterior, se analiza la proporción de publicaciones `reliable`
#  y `unreliable` agrupando por categoría. A diferencia de los idiomas, hay gran
#  disparidad en la calidad de las etiquetas siendo las menos confiables más
#  abundantes. En particular, para ninguna categoría la proporción de `reliables`
#  es mayor a 22.2945%. Otras categorías como `WINES` la cantidad de etiquetas
#  confiables es incluso menor al 3%.

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
df_label_by_category.sort_values(by="proportion")


# %% [markdown]
#  Nuevamente para el total de publicaciones, solamente el 94882 (14.67%) son
#  `reliable` y 551878 (85.32%) son `unreliable`.

# %%
nof_items = len(df)
nof_items

# %%
nof_reliable_labels = len(df[df[label_quality] == "reliable"])
(nof_reliable_labels, nof_reliable_labels / nof_items * 100)

# %%
nof_unreliable_labels = nof_items - nof_reliable_labels
(nof_unreliable_labels, nof_unreliable_labels / nof_items * 100)

# %% [markdown]
#  ## Relación entre el label quality y el idioma
#  La confiabilidad de las etiquetas podría deberse a si están en español o
#  portugués, por ello se optó por calcular la proporción de estas pero agrupadas
#  por idioma.

# %%
df_label_by_language = proportion(df, language, label_quality)
df_label_by_language

# %% [markdown]
#  Las proporciones de `reliable` para portugués y para español son parecidas, de
#  un 15% y un 13% respectivamente. Recordar además que del total de
#  publicaciones el 14,67 % son `reliable`. Esto hace que pensar que hay
#  independencia entre las variables `language` y `label_quality`. Si se toman
#  los cantidades de la columna `count`, tenemos cuatro casos posibles:
#

# %%
pd.crosstab(
    df[language],
    df[label_quality]
)

# %% [markdown]
#  Las variables aleatorias $X_{A}$ y $X_{B}$ asociadas a las poblaciones
#  `language` y `label_quality` son independientes si para cada $x_{A} \in
#  \{spanish, portugues\}$ y $x_{B} \in \{reliable, unreliable\}$ valores de
#  ambas poblaciones, se tiene que:
#
#  $P(X_{A}=x_{A}, X_{B}=x_{B}) =
#  P(X_{A}=x_{A})P(X_{B}=x_{B})$
#
#  A continuación, para cada uno de los cuatro casos se calcula el cociente entre
#  esas probabilidades. Para ello, se considera la cantidad publicaciones en la muestra
#  que adoptan el valor $x_A$, `nof_A`

# %%
nof_items = len(df)
nof_A = df.groupby(language).size()
nof_A

# %% [markdown]
#  Similarmente la cantidad de publicaciones que adoptan el valor $x_B$, `nof_B`

# %%
nof_B = df.groupby(label_quality).size()
nof_B

# %% [markdown]
#  Luego aquellas que coinciden en $x_A$ y $x_B$, `nof_AB`

# %%
nof_AB = df_label_by_language["count"]
nof_AB

# %% [markdown]
#  Luego se divide entre la cantidad de publicaciones para finalmente obtener

# %%
# First divide between @nof_B and @nof_AB * nof_items so a dataframe of shape
# @nof_AB is obtained
nof_A * (nof_B / (nof_AB * nof_items))

# %% [markdown]
#  Como los valores son todos muy cercanos a 1, se puede concluir que `language` y
#  `label_quality` son independientes.
#
#  Se hará un gráfico de barras, donde en lugar de mostrar el total de
#  publicaciones para `reliable` y `unreliable` por idioma, se exhibirá la
#  `proporción`, para español, para portugués, y para ambos idiomas.

# %%
nof_reliables = len(df[df[label_quality] =="reliable"])
nof_items = len(df)

df_label_by_language = df_label_by_language.reset_index()
cols = list(df_label_by_language.columns)

both_reliable=[
    "both",
    "reliable",
    nof_reliables,
    nof_items,
    nof_reliables/nof_items
]
both_unreliable=[
    "both",
    "unreliable",
    nof_items - nof_reliables,
    nof_items,
    (nof_items - nof_reliables)/nof_items
]

# %%
df_label_by_language = df_label_by_language.append(
    {label:value for label, value in zip(cols, both_reliable)},
    ignore_index=True
)
df_label_by_language = df_label_by_language.append(
    {label:value for label, value in zip(cols, both_unreliable)},
    ignore_index=True
)
df_label_by_language

# %%
fig = plt.figure(figsize=(10,6))
seaborn.barplot(
    y=df_label_by_language["proportion"],
    x=df_label_by_language["language"],
    hue=df_label_by_language["label_quality"],
    ci=None
)
plt.xticks(rotation=30)
plt.ylabel("Proporción Reliable vs Unreliable")
plt.xlabel("Idioma")
plt.ticklabel_format(style='plain', axis='y')

# %% [markdown]
#  ## Exploración de la estructura de los títulos
#  Para obtener una medida cuantitativa de la estructura que tiene el título de
#  los articulos se optó por contabilizar por la cantidad de palabras, stopwords,
#  dígitos, y caracteres especiales que aparecen en este por medio de las
#  funciones `count_words`, `count_stopwords`, `count_digits`,
#  `count_special_chars` aplicandose sobre cada item. (Puede tardar unos segundos
#  debido al tamaño de la base de datos y a las operaciones que se realizan)

# %%
df_analysis_of_words = df[title] \
    .agg([
        count_words,
        count_stopwords,
        count_digits,
        count_special_chars
    ]).join(df[[title, category]])
df_analysis_of_words

# %%
df_analysis_of_words.describe()

# %% [markdown]
#  Notar que se tiene en promedio un total de 7.52 palabras por título. En cuanto
#  a stopwords, a pesar de ser recurrentes en textos como conectivos, este no fue
#  el caso, probablemente debido a la necesidad de hacer énfasis en el item a
#  vender en la menor cantidad de palabras posibles. De manera similar ocurre con
#  la cantidad de caracteres especiales usados. Sin embargo, se dan algunos
#  valores atípicos como el uso de 38 caracteres especiales en una publicación.

# %% [markdown]
#  Para conocer la cantidad de palabras con las que se cuenta por categoría se
#  utiliza el data frame anterior para calcular el promedio de ellas.

# %%
df_analysis_of_words.groupby(category).describe()

# %%
fig = plt.figure(figsize=(12,8))
seaborn.barplot(
    data=df_analysis_of_words.groupby(category).mean().reset_index(),
    x=category,
    y="count_words",
    color="salmon"
)
plt.xticks(rotation=80)

# %% [markdown]
#  Nuevamente no hay una disparidad fuerte entre la cantidad de palabras usadas
#  en el título por las categoría.

# %% [markdown]
#  Para concluir el análisis se decidió considerar no solo la cantidad de
#  palabras si no también la frecuencia en la que aparecen y si corresponden o
#  tienen sentido con la categoría a la que fueron asociadas. Para ello, se
#  remueven las stopwords, y los caracteres numéricos y especiales por medio de
#  la función `remove_unimportant_words`. Luego para cada categoría se obtiene la
#  frecuencia de todas las palabras que hayan aparecido en sus títulos para
#  finalmente quedarse con las mejores 10. Algo a aclarar es que no se están
#  aplicando técnicas de lematización al título si no que directamente se
#  trabajan con su representación en minúsculas lo cual podría dar lugar a
#  palabras que tienen una conjugación similar. Tampoco se está separando por
#  palabras que correspondan solamente al español o al portugués llevando así a
#  dar lugar a palabras frecuentes que tengan misma traducción. El siguiente data
#  frame muestra estos resultados (La operación puede llevar unos
#  segundos).

# %%
df_word_freq = df[title]\
    .apply(remove_unimportant_words) \
    .to_frame() \
    .join(df[category]) \
    .groupby(category) \
    .agg(" ".join)[title] \
    .apply(lambda s: nltk.FreqDist(word_tokenize(s)).most_common(10)) \
    .to_frame() \
    .rename(columns={"title":"top10_word_freq"}) \
    .reset_index()
df_word_freq

# %% [markdown]
#  Finalmente, si se elige alguna categoría en particular, por ejemplo
#  `BABY_CAR_SEATS`, podemos obtener el siguiente gráfico de frecuencia.

# %%
all_fdist = pd.Series(dict(
    df_word_freq[
        df_word_freq["category"] == "BABY_CAR_SEATS"
    ]["top10_word_freq"][0])
)
plt.figure(figsize=(20, 10))
seaborn.barplot(
    x=all_fdist.index,
    y=all_fdist.values
)
plt.xticks(rotation=30)

# %% [markdown]
#  Claramente palabras como *auto*, *kg*, *butaca*, *bebe*, *huevito*, etc,
#  corresponden con las esperadas para la compra de asientos para bebé con una
#  frecuencia superior a las 4000 para las que aparecen en el gráfico. Por otro
#  lado notar como unidades de medida tales como el *kg* son también útiles para
#  este tipo de compras y que salieron a la vista dado por la eliminación de
#  dígitos.

# %% [markdown]
# ## Conclusión
# A pesar de tener una cantidad reducida de variables aleatorias a considerar en
# el conjunto de datos, la exploración deja en evidencia una basta cantidad de
# aspectos a analizar por medio de la contabilización de palabras con ciertas
# propiedades. Se dejó en claro que la ocurrencia de stopwords, dígitos y
# caracteres especiales a pesar de ser poco frecuentes, palabras cercanas a ellas
# pueden ser de gran importancia en la publicación de un artículo. También la
# independencia entre las variables `language` y `label_quality` es de gran
# valor ya que con ello se podrían realizar test de hipótesis bajo estadísticos
# que requieren esta suposición y que serían ideales para un futuro análisis.
# Lo mismo sucede con la lematización del título y la consideración de frecuencias
# por idioma. Se piensa que con ello se podrían determinar otras palabras relevantes 
# que no aparecieron en esta primera exploración.
#
# Otros aspectos importantes del conjunto de datos son la mayor cantidad de etiquetas
# poco confiables para la categorización de las publicaciones y la poca frecuencia
# de caracteres no relevantes en los títulos.

# %%
