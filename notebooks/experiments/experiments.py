# %% [markdown]
"""
## Pipeline
Con el fin de automatizar la ejecución de los modelos que se van utilizar para
experimentar y proponer hipótesis, se optó por implementar un pipeline que realicen
de principio a fin

- Carga de datos (`load_data`)
- Preprocesamiento de datos (`preprocess_data`)
- Split de datos (`split_data`)
- Códificación de titulos, confección de word embeddings (`encode_data`)
- Selección de modelos predefinidos (`select_model`)
- Partición en train y validación. Obtención de métricas (`k_fold_cross_validation`)
- Evaluación el modelo. Obtención de métricas (`evaluate_model`)
- Guardar predicciones

Para el registro de métricas se hizo uso de `mlflow` una libreria para la organización
del flujo de trabajo en procesos de *Machine Learning* permitiendo además
realizar automaticamente las curvas de accuracy y loss.
Se entrenaron 3 modelos:

- base: Modelo baseline de dos capas ocultas de 256 y 128 respectivamente.
- base_wd: Modelo baseline con dropout entre las capas ocultas de 0.4.
- base_wbn: Modelo baseline con batch normalization entre las capas ocultas.

La siguientes celdas son un ejemplo de ejecución de algunos experimentos con
estos modelos utilizando custom word embeddings y una muestra del dataframe
completo para evitar realizar el entrenamiento completo.
"""
# %%
import mlflow
from mlflow.tracking import MlflowClient
from dataclasses import dataclass
from pipeline import Pipeline
# %%
@dataclass
class Params:
    model:str
    kfolds = 5
    seed = 0
    batch_size = 128
    embedding_dim = 50
    embedding_url = "https://www.famaf.unc.edu.ar/~nocampo043/fasttext_sp_pt.vec"
    embedding_type = "custom"
    nof_samples = 20000
    test_size = 0.2
    epochs = 20
    dataset_url = "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"

# %%
params = Params(model="base")
pipeline = Pipeline(params)
# %% [markdown]
"""
## Carga de datos (`load_data`)
"""
# %%
pipeline.load_data()
# %%
pipeline.dataset
# %% [markdown]
"""
## Preprocesamiento de datos (`preprocess_data`)
"""
# %%
pipeline.preprocess_data()
# %%
pipeline.x
# %%
pipeline.y
# %% [markdown]
"""
## Split de datos (`split_data`)
"""
# %%
pipeline.split_data()
# %%
pipeline.x_train
# %%
pipeline.y_train
# %%
pipeline.x_test
# %%
pipeline.y_test
# %% [markdown]
"""
## Códificación de titulos, confección de word embeddings (`encode_data`)
"""
# %%
pipeline.encode_data()
# %%
pipeline.vectorize_layer
# %%
pipeline.vocab_size
# %%
pipeline.y_train
# %%
pipeline.y_test
# %% [markdown]
"""
## Selección de modelos predefinidos (`select_model`)
"""
# %%
pipeline.select_model()
# %%
pipeline.model.summary()
# %% [markdown]
"""
## Partición en train y validación. Obtención de métricas (`k_fold_cross_validation`)
"""
# %%
pipeline.k_fold_cross_validation()
# %% [markdown]
"""
## Evaluación el modelo. Obtención de métricas (`evaluate_model`)
"""
# %%
pipeline.evaluate_model()
# %%
# %% [markdown]
"""
## Guardar predicciones (`save_predictions`)
"""
pipeline.save_predictions()
# %% [markdown]
"""
## Mlflow
"""
# %%
def run_model(model_name):
    params = Params(model=model_name)
    client = MlflowClient()

    try:
        experiment_id = client.create_experiment(model_name)
    except:
        experiment_id = client.get_experiment_by_name(model_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id):
        pipeline = Pipeline(params)

        pipeline.run()

# %%
run_model(model_name="base")
# %%
run_model(model_name="base_wd")
#%%
run_model(model_name="base_wbn")

# %% [markdown]
"""
Para visualizar las métricas de cada uno de los experimentos es necesario ejecutar
"""
# %%
!mlflow ui
# %% [markdown]
"""
Entre los resultados obtenidos a priori nuestro modelo baseline parece obtener
los mejores resultados en testing. Sin embargo, no se realizaron busquedas
exhaustivas de modelos o hiperparámetros sobre los otros modelos, para
intentar obtener alguna mejora sustancial que supere a nuestro baseline. Se
trabajó especialmente en la infraestructura y layout para poder ejecutar
automáticamente futuros experimentos.
"""