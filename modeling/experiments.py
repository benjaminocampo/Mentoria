# %% [markdown]
# # Categorización de publicaciones de productos de Mercado Libre
# Autores: Maximiliano Tejerina, Eduardo Barseghian, Benjamín Ocampo
# %% [markdown]
# Una vez finalizadas las etapas de visualización de datos, preprocesamiento, y
# codificación, sobre el conjunto de datos dado por el ML Challenge 2019, se
# almacenó dicho dataset de manera remota para facilitar su acceso durante los
# experimentos que se trabajaron durante esta sección.
#
# Las modificaciones y decisiones de diseño tomadas durante las etapas
# anteriores pueden encontrarse en los directorios `exploration`, y `encoding`.
# %% [markdown]
# ## Reproducción en Google Colab
# Esta notebook puede ejecutarse de manera local utilizando un entorno de
# `conda` por medio del archivo de configuración `environment.yml` ubicado en la
# raíz del proyecto o bien de manera online por medio de `Google Colab`. Para
# este último, es necesario ejecutar la siguiente celda con las dependencias
# necesarias e incluir los módulos que se encuentran en este directorio.
# %%
# !pip install mlflow
# !pip install keras
# %% [markdown]
# ## Pipeline
# Dado que el objetivo de este proyecto es estimar la categoría a la cual
# pertenece un título de una publicación de Mercado Libre. Se desarrolló un
# *pipeline* de ejecución a partir del conjunto de datos preprocesado.
#
# ![Pipeline](images/pipeline.png)
#
# Las capas de vectorización y embedding fueron llevadas a fondo en las
# secciones preprocesamiento y codificación, permitiendo proyectar los títulos
# de las publicaciones en un espacio N dimensional que preserva la semántica de
# las palabras.
#
# El foco de esta sección, que denominamos `modeling`, consiste en encontrar el
# modelo o clasificador que obtenga el mejor `balanced_accuracy` en la
# clasificación de las publicaciones. Dicha métrica es la que nos interesa mejorar
# durante la competencia y será relevante durante la búsqueda de
# hiperparametros.
#
# La ejecución de uno o más procesos en este *pipeline* es lo que definiremos
# como un experimento, donde propondremos como hipótesis una configuración sobre
# el segundo, tercer, y cuarto paso. Finalmente, el registro de resultados se
# llevó a cabo de la librería mlflow sobre el último paso.
#
# A su vez, varias funciones *helper* fueron definidas en respectivos archivos
# para facilitar la implementación del *pipeline*.
#
# Estos se disponen en:
#
# - `models.py`: Definición de la arquitectura de las redes neuronales usadas.
# - `embedding.py`: Generación de *in-domain embeddings* y extracción de
#   codificaciones *pre-trained*.
# - `preprocess.py`: Herramientas de preprocesamiento de texto para los títulos
#   del *dataset*.
# - `encoding.py`: Codificación de títulos y etiquetas.
# %% [markdown]
# ## Librerias
# %%
# Pandas
import pandas as pd
# Seaborn
import seaborn as sns
# Matplotlib
import matplotlib.pyplot as plt
# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
# Utils
from transformers import FastTextVectorizer
from models import FeedForward_Net, LSTM_Net
# %% [markdown]
# ## Sampling de datos
# Debido al gran tamaño de muestras disponibles (por encima de los 600000
# ejemplares), se optó por considerar únicamente un subconjunto aleatorio del
# mismo para realizar los experimentos.
# %%
URL = "https://www.famaf.unc.edu.ar/~nocampo043/ML_2019_challenge_dataset_preprocessed.csv"
NOF_SAMPLES = 20000
SEED = 0

df = pd.read_csv(URL)
df = df.sample(n=NOF_SAMPLES, random_state=SEED)
df
# %% [markdown]
# ## Train-Test Split
# Durante la separación en conjuntos de *train* y *test*, se definió
# inicialmente la variable objetivo `y = "encoded_category"` sin especificar las
# características. Esto fue para mantener la estructura de datos en la que se
# encuentran almacenados los ejemplares, y se permita filtrar de manera sencilla
# los conjuntos de *train* y de *test*, por *label_quality*. De esta manera, se
# pudo discriminar el *balanced_accuracy_score* en *test* para estos dos
# subconjuntos.
# %%
y = df["encoded_category"]
df_train, df_test, y_train, y_test = train_test_split(df,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=SEED)

df_test_unrel = df_test.loc[df_test["label_quality"] == "unreliable"]
df_test_rel = df_test[df_test["label_quality"] == "reliable"]
# %%
y_test_unrel = df_test_unrel["encoded_category"]
y_test_rel = df_test_rel["encoded_category"]
# %% [markdown]
# Una vez hecho esto, las *features* se obtienen proyectando la columna
# `cleaned_title` de estos *dataframes*.
# %%
x_train = df_train["cleaned_title"]
x_test = df_test["cleaned_title"]
x_test_rel = df_test_rel["cleaned_title"]
x_test_unrel = df_test_unrel["cleaned_title"]

length_longest_sentence = int(df["cleaned_title"].apply(lambda title: len(title.split())).max())
# %% [markdown]
# ## 
# El experimento a evaluar en esta notebook consta de la comparación de tres
# modelos:
#
# - LogisticRegression (lgr)
# - Red LSTM (lstm)
# - Red Feed Forward (ff)
#
# Para la busqueda de parámetros se utilizó una `Grid Search CV` bajo el mismo
# espacio de busqueda en las redes neuronales
# %% [markdown]
# ## Baseline
vector_dim = 100
transformer = FastTextVectorizer(dim=vector_dim, return_sequences=False)
transformer.fit(x_train)
x_train_arr = transformer.transform(x_train)
x_test_arr = transformer.transform(x_test)
x_test_rel_arr = transformer.transform(x_test_rel)
x_test_unrel_arr = transformer.transform(x_test_unrel)
# %%
model = LogisticRegression()
param_grid = {
    "C": [1, 0.1, 0.001],
    "penalty": ["elasticnet"],
    "class_weight": ["balanced"],
    "random_state": [0],
    "solver": ["saga"],
    "l1_ratio": [0, 1, 0.5]
}
# %%
clf = GridSearchCV(estimator=model,
                   param_grid=param_grid,
                   cv=5,
                   scoring="balanced_accuracy",
                   verbose=3)
clf.fit(x_train_arr, y_train)
# %%
clf.best_estimator_.predict(x_test_arr)
y_pred = clf.best_estimator_.predict(x_test_arr)
y_pred_rel = clf.best_estimator_.predict(x_test_rel_arr)
y_pred_unrel = clf.best_estimator_.predict(x_test_unrel_arr)
blc_acc = balanced_accuracy_score(y_test, y_pred)
blc_acc_rel = balanced_accuracy_score(y_test_rel, y_pred_rel)
blc_acc_unrel = balanced_accuracy_score(y_test_unrel, y_pred_unrel)
# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7), dpi=600)
cm = sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Regresión Logística")
plt.savefig("LogisticRegression.png")
plt.clf()
# %%
clf.cv_results_["best_blc_acc"] = blc_acc
clf.cv_results_["best_blc_acc_rel"] = blc_acc_rel
clf.cv_results_["best_blc_acc_unrel"] = blc_acc_unrel
results = pd.DataFrame(clf.cv_results_)
results.to_csv("LogisticRegression.csv", index=False)
# %% [markdown]
# ## NNet FeedForward
# %%
ff_net = FeedForward_Net(vector_dim=vector_dim)

param_grid = {
    "batch_size": [100],
    "epochs": [100],
    "units": [256],
    "lr": [1e-1],
    "dropout": [.4],
}
clf = GridSearchCV(estimator=ff_net,
                   param_grid=param_grid,
                   cv=5,
                   scoring="balanced_accuracy",
                   verbose=3)
clf.fit(x_train_arr, y_train)
# %%
y_pred = clf.best_estimator_.predict(x_test_arr)
y_pred_rel = clf.best_estimator_.predict(x_test_rel_arr)
y_pred_unrel = clf.best_estimator_.predict(x_test_unrel_arr)
blc_acc = balanced_accuracy_score(y_test, y_pred)
blc_acc_rel = balanced_accuracy_score(y_test_rel, y_pred_rel)
blc_acc_unrel = balanced_accuracy_score(y_test_unrel, y_pred_unrel)
# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7), dpi=600)
cm = sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Red Feed Forward")
plt.savefig("Feed_Forward.png")
plt.clf()
# %%
clf.cv_results_["best_blc_acc"] = blc_acc
clf.cv_results_["best_blc_acc_rel"] = blc_acc_rel
clf.cv_results_["best_blc_acc_unrel"] = blc_acc_unrel
results = pd.DataFrame(clf.cv_results_)
results.to_csv("FeedForward.csv", index=False)
# %% [markdown]
# ## Sequential LSTM Net
# %%
vector_dim = 100
lstm_net = LSTM_Net(vector_dim=vector_dim,
                    sequence_dim=length_longest_sentence,
                    batch_size=100,
                    epochs=100,
                    units=256,
                    lr=1e-1,
                    dropout=.1)
transformer = FastTextVectorizer(dim=vector_dim,
                                 length_longest_sentence=length_longest_sentence,
                                 return_sequences=True)
transformer.fit(x_train)
x_train_arr = transformer.transform(x_train)
x_test_arr = transformer.transform(x_test)
x_test_rel_arr = transformer.transform(x_test_rel)
x_test_unrel_arr = transformer.transform(x_test_unrel)
# %%
lstm_net.fit(x_train_arr, y_train)
# %%
y_pred = lstm_net.predict(x_test_arr)
y_pred_rel = lstm_net.predict(x_test_rel_arr)
y_pred_unrel = lstm_net.predict(x_test_unrel_arr)
blc_acc = balanced_accuracy_score(y_test, y_pred)
blc_acc_rel = balanced_accuracy_score(y_test_rel, y_pred_rel)
blc_acc_unrel = balanced_accuracy_score(y_test_unrel, y_pred_unrel)
# %%
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7), dpi=600)
cm = sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Red LSTM")
plt.savefig("LSTM.png")
plt.clf()
# %% [markdown]
# Para una cantidad de ejemplares de 20000, se obtuvo muy buenos resultados con
# la red Feed Forward y nuestro Baseline en balanced_accuracy en los conjuntos
# reliable, unreliable, y sin filtro alguno. Nos sorprendio el resultado de la
# LSTM ya que supusimos que podía llegar a tomar ventaja de la secuencia de los
# títulos pero no pudimos lograr que aprenda el problema. Se cree que utilizando
# todos los datos podría llegar a mejorar los resultados de este modelo.
