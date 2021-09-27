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
# !pip install gensim --upgrade
# !pip install scikit-learn==0.21.2
# !pip install numpy==0.19
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
# MLflow
import mlflow
# Pandas
import pandas as pd
# Keras
from keras.wrappers.scikit_learn import KerasClassifier
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score
# Scipy
from scipy.stats import uniform, randint, loguniform
# Utils
from models import ff_model, lstm_model
# TODO: KerasClassifier internal implementation uses .predict() when searching
# hyperparameters. Since this utility is deprecated, this warning is displayed
# during tuning.
import warnings
warnings.filterwarnings("module", category=DeprecationWarning)
# %% [markdown]
# ## Sampling de datos
# Debido al gran tamaño de muestras disponibles (por encima de los 600000
# ejemplares), se optó por considerar únicamente un subconjunto aleatorio del
# mismo para realizar los experimentos.
# %%
URL = "https://www.famaf.unc.edu.ar/~nocampo043/ML_2019_challenge_dataset_preprocessed.csv"
NOF_SAMPLES = 2000
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
# %% [markdown]
# ## Padding
# Dado que en los títulos de las publicaciones la cantidad de palabras que
# ocurren varía, es necesario extender los vectores representantes a un ancho
# común, en este caso, el de la oración más larga.
# %%
length_long_sentence = (df["cleaned_title"].apply(lambda s: s.split()).apply(
    lambda s: len(s)).max())
# %% [markdown]
# ## LSTM vs Feed Forward (lstm_vs_ff)
# El experimento a evaluar en esta notebook consta de la comparación de dos
# modelos:
# - Red LSTM (lstm)
# - Red Feed Forward (ff)
#
# Ambos cuentan con capas de vectorización, *embedding*, aprendizaje, *dropout*,
# y predicción diferenciandose en la de aprendizaje. Estas son una `LSTM layer`
# y una `Dense layer` respectivamente.
#
# Para la busqueda de parámetros se utilizó una `Randomized Search CV` bajo las
# mismas distribuciones en ambos modelos.
# %%
mlflow.set_experiment("LSTM vs Feed Forward")

with mlflow.start_run():
    build_ff = lambda units, dropout, lr, embedding_dim: ff_model(
        x_train,
        length_long_sentence,
        units,
        dropout,
        lr,
        embedding_dim)

    build_lstm = lambda units, dropout, lr, embedding_dim: lstm_model(
        x_train, length_long_sentence, units, dropout, lr, embedding_dim)

    ff = KerasClassifier(build_fn=build_ff)

    lstm = KerasClassifier(build_fn=build_lstm)

    dist = {
        "batch_size": [100, 1000],
        "epochs": [5, 10],
        "units": randint(256, 512),
        "lr": loguniform(1e-4, 1e-1),
        "dropout": uniform(.1, .6),
        "embedding_dim": randint(50, 200)
    }

    searches = [
        ("lstm", lstm, dist),
        ("ff", ff, dist)
    ]

    for model_name, hyper_model, dist in searches:
        mlflow.log_params(dist)

        clf = RandomizedSearchCV(estimator=hyper_model,
                                 param_distributions=dist,
                                 cv=5,
                                 scoring="balanced_accuracy",
                                 verbose=3)

        clf.fit(x_train, y_train)

        y_pred = clf.best_estimator_.predict(x_test)
        y_pred_rel = clf.best_estimator_.predict(x_test_rel)
        y_pred_unrel = clf.best_estimator_.predict(x_test_unrel)

        blc_acc = balanced_accuracy_score(y_test, y_pred)
        blc_acc_rel = balanced_accuracy_score(y_test_rel, y_pred_rel)
        blc_acc_unrel = balanced_accuracy_score(y_test_unrel, y_pred_unrel)

        acc = accuracy_score(y_test, y_pred)
        acc_rel = accuracy_score(y_test_rel, y_pred_rel)
        acc_unrel = accuracy_score(y_test_unrel, y_pred_unrel)

        mlflow.log_metric("balanced_accuracy", blc_acc)
        mlflow.log_metric("balanced_accuracy reliable", blc_acc_rel)
        mlflow.log_metric("balanced_accuracy unreliable", blc_acc_unrel)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("accuracy reliable", acc_rel)
        mlflow.log_metric("accuracy unreliable", acc_unrel)

        predictions = pd.DataFrame(data={"y_pred": y_pred, "y_test": y_test})
        predictions.to_csv(f"{model_name}_predictions.csv", index=False)
# %%
!mlflow ui
# %% [markdown]
# Para una cantidad de ejemplares de 10000, se obtuvo mejores resultados con la red Feed Forward
# en balanced_accuracy en los conjuntos reliable, unreliable, y sin filtro alguno.
# Con aproximadamente 0.70 para la `ff` y de 0.40 para `lstm`. Estos resultados probablemente
# se hayan dado debido al muestreo donde algunas de las clases de no fue vista durante entrenamiento.
# A su vez, debido a la capacidad de las redes lstm de poder visualizar en mayor medida
# el contexto en una oración, se esperaría tener mejores resultados alterando la arquitectura de la red.
