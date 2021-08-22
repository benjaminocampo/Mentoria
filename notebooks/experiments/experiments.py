#%%
import mlflow
from mlflow.tracking import MlflowClient
from dataclasses import dataclass
from pipeline import Pipeline

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