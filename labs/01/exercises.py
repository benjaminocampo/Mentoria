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

    df_proportion["proportion"] = df_proportion["count"] / df_proportion["total"]
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
# %% [markdown]
# ## Publicaciones de items dentro de cada categoría

# %%

df.groupby(category).size()

# %% [markdown]
# ## Proporción de publicaciones en español y portugués dentro de cada categoría

# %%
proportion(df, category, language)
# %% [markdown]
# ## Proporción de label quality dentro de cada categoría
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
    .aggregate({
        "nof_stopwords": count_stopwords,
        "nof_digits": count_digits,
        "nof_special_chars": count_special_chars
    })