#%%
import mlflow
from mlflow.tracking import MlflowClient
from dataclasses import dataclass
from pipeline import Pipeline

@dataclass
class Params:
    kfolds = 5
    seed = 0
    batch_size = 128
    embedding_dim = 50
    embedding_url = "https://www.famaf.unc.edu.ar/~nocampo043/fasttext_sp_pt.vec"
    embedding_type = "custom"
    nof_samples = 5000
    test_size = 0.2
    epochs = 5
    dataset_url = "https://www.famaf.unc.edu.ar/~nocampo043/ml_challenge2019_dataset.csv"
# %%
params = Params()

client = MlflowClient()

# If the project does not exists, it creates a new one
# If the project already exists, it is taken the project id
try:
    # Creates a new experiment
    experiment_id = client.create_experiment("baseline")

except:
    # Retrieves the experiment id from the already created project
    experiment_id = client.get_experiment_by_name("baseline").experiment_id

# Initialize mlflow context
with mlflow.start_run():
    # Run pipeline
    pipeline = Pipeline(params)
    pipeline.run()
# %%
