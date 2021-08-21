# Tqdm
from tqdm import tqdm
# Data
import numpy as np
import pandas as pd
# MLflow
import mlflow
from mlflow.tracking import MlflowClient
# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
# Utilities
from models import load_embedding, create_embedding_layer, baseline_model
from preprocess import clean_text
from encoding import encode_labels, tokenize_features
from parser import get_parser


class Pipeline:
    def __init__(self, params):
        self.params = params
        self.dataset_path = params.dataset_path
        self.dataset = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.model = None
        self.embeddings = None
        self.vocab = None

    def load_data(self):
        # Read dataset
        self.dataset = pd.read_csv(self.dataset_path)

        self.dataset = self.dataset.sample(5000, random_state=self.params.seed)

    def preprocess_data(self):
        # Remove numbers, symbols, special chars, contractions, etc.
        cleaned_title_col = self.dataset[["title", "language"
                                          ]].apply(lambda x: clean_text(*x),
                                                   axis=1)

        # Add @cleaned_title column to dataset
        self.dataset = self.dataset.assign(cleaned_title=cleaned_title_col)

        # Set titles and categories as features and labels
        self.x = self.dataset["cleaned_title"]
        self.y = self.dataset["category"]

    def split_data(self):
        # Split dataset into training and testing instances
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=self.params.test_size,
            random_state=self.params.seed)

    def encode_data(self):
        # Encode training and testing titles separately
        self.x_train, self.x_test, self.vocab = tokenize_features(
            self.x_train, self.x_test)

        # Encode training and testing labels
        self.y_train, self.y_test = encode_labels(self.y_train, self.y_test)

        # Load pre-trained embeddings
        self.embeddings, nof_hits, nof_misses = load_embedding(
            self.params.embedding_file, self.vocab, self.params.embedding_dim)

        # Track the number of words of the vocab that appeared in the
        # pre-trained embedding
        mlflow.log_metric("nof hits word emb", nof_hits)
        mlflow.log_metric("nof misses word emb", nof_misses)

    def select_model(self):
        # Create embedding layer
        embedding_layer = create_embedding_layer(
            vocab_size=len(self.vocab) + 1,
            embedding_dim=self.params.embedding_dim,
            embedding_matrix=self.embeddings)

        # Choose model
        # TODO: Here we should choose over a variaty of models. They could also
        # be implemented using classes
        self.model = baseline_model(embedding_layer=embedding_layer,
                                    nof_classes=len(np.unique(self.y_train)))

    def k_fold_cross_validation(self):
        # Log parameters
        mlflow.log_param("seed", self.params.seed)
        mlflow.log_param("kfolds", self.params.kfolds)
        mlflow.log_param("epochs", self.params.epochs)
        mlflow.log_param("batch size", self.params.batch_size)

        initial_weigths = self.model.get_weights()
        val_accuracy = []
        val_loss = []
        # Split on training to get cross-validation sets
        skf = StratifiedKFold(n_splits=self.params.kfolds, shuffle=False)
        for fold_id, (train_indices, val_indices) in enumerate(
                skf.split(self.x_train, self.y_train)):
            # Fit the model and get the history of improvement
            history = self.model.fit(self.x_train[train_indices],
                                     self.y_train[train_indices],
                                     batch_size=self.params.batch_size,
                                     epochs=self.params.epochs,
                                     verbose=1)
            # Evaluate in validation data
            loss, accuracy = self.model.evaluate(
                self.x_train[val_indices],
                self.y_train[val_indices],
                batch_size=self.params.batch_size,
                verbose=1)

            val_accuracy.append(accuracy)
            val_loss.append(loss)

            for epoch in range(self.params.epochs):
                mlflow.log_metric(f"fold {fold_id} accuracy curve",
                                  history.history["accuracy"][epoch],
                                  step=epoch)
                mlflow.log_metric(f"fold {fold_id} loss curve",
                                  history.history["loss"][epoch],
                                  step=epoch)


            # Predecir los y's usando self.model, pasandoles x_train 

            # Reset model for next iteration
            self.model.set_weights(initial_weigths)
        mlflow.log_metric("mean val accuracy", np.mean(val_accuracy))
        mlflow.log_metric("mean val loss", np.mean(val_loss))

    def evaluate_model(self):
        # Train model again but this time using all training data
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=self.params.batch_size,
                                 epochs=self.params.epochs,
                                 verbose=1)
        loss, accuracy = self.model.evaluate(self.x_test,
                                             self.y_test,
                                             batch_size=self.params.batch_size,
                                             verbose=1)

        for epoch in range(self.params.epochs):
            mlflow.log_metric(f"test accuracy curve",
                              history.history["accuracy"][epoch],
                              step=epoch)
            mlflow.log_metric(f"test loss curve",
                              history.history["loss"][epoch],
                              step=epoch)

        mlflow.log_metric("test accuracy", accuracy)
        mlflow.log_metric("test loss", loss)

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.encode_data()
        self.select_model()
        self.k_fold_cross_validation()
        self.evaluate_model()


if __name__ == '__main__':
    parser = get_parser()

    params = parser.parse_args()

    # Init Mlflow client
    client = MlflowClient()

    # Creates a new experiment
    experiment_id = client.create_experiment(params.experiment_name)

    # Initialize mlflow context
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=params.experiment_name):
        # Run pipeline
        pipeline = Pipeline(params)
        pipeline.run()