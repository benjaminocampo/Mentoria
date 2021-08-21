# Tqdm
from tqdm import tqdm
# Pandas
import pandas as pd
# MLflow
import mlflow
from mlflow.tracking import MlflowClient
# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
# Utilities
from models import load_embedding, create_embedding_layer, create_baseline_model
from preprocess import clean_text
from encoding import encode_labels, tokenize_features
from graphs import plot_accuracy_curve
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

    def preprocess_data(self):
         # Activate progress bar in pandas
        tqdm.pandas()

        # Remove numbers, symbols, special chars, contractions, etc.
        cleaned_title_col = self.dataset[["title", "language"]].progress_apply(
            lambda x: clean_text(*x), axis=1)

        # Add @cleaned_title column to dataset
        self.dataset = self.dataset.apply(cleaned_title=cleaned_title_col)

        # Set titles and categories as features and labels
        self.x = self.dataset["cleaned_title"]
        self.y = self.dataset["category"]

    def split_data(self):
        # Split dataset into training and testing instances
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=self.params.test_size)

    def encode_data(self):
        # Encode training and testing instances separately
        self.x_train, self.x_test, self.vocab = tokenize_features(
            self.x_train, self.x_test)

        # Encode training and testing labels
        self.y_train, self.y_test = encode_labels(self.y_train, self.y_test)

        # Load pre-trained embeddings
        self.embeddings = load_embedding(self.params.embedding_file, self.vocab,
                                         self.params.embedding_dim)

    def select_model(self):
        # Create embedding layer
        embedding_layer = create_embedding_layer(
            vocab_size=len(self.vocab) + 1,
            embedding_dim=self.params.embedding_dim,
            embedding_matrix=self.embeddings)

        # Choose model
        # TODO: Here we should choose over a variaty of models. They could also
        # be implemented using classes
        model = create_baseline_model(embedding_layer=embedding_layer,
                                      nof_classes=len(self.y_train.unique()))
        self.model = model

    def k_fold_cross_validation(self):
        # Split on training to get cross-validation sets
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        for train_indices, val_indices in skf.split(self.x_train,
                                                    self.y_train):
            # Fit the model and get the history of improvement
            history = self.model.fit(self.x_train[train_indices],
                                     self.y_train[train_indices],
                                     batch_size=128,
                                     epochs=5,
                                     verbose=1)
            # Evaluate in validation data
            loss, accuracy = self.model.evaluate(self.x_train[val_indices],
                                                 self.y_train[val_indices],
                                                 batch_size=128,
                                                 verbose=0)

            # Get accuracy plot
            fig = plot_accuracy_curve(history.history["accuracy"])

            # Save accuracy curve
            fig.savefig(
                f"{self.params.img_path}/{self.params.experiment_name}.png")

            # Track images and metrics
            mlflow.log_artifact(
                f"{self.params.img_path}/{self.params.experiment_name}.png")
            mlflow.log_params()
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("loss", loss)

    def evaluate_model(self):
        loss, accuracy = self.model.evaluate(self.x_train,
                                             self.y_test,
                                             batch_size=128,
                                             verbose=0)

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