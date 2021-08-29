# Data
import numpy as np
import pandas as pd
# MLflowf
import mlflow
# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
# Utilities
from models import (load_embedding, create_embedding_layer, baseline_model,
                    baseline_with_dropout_model, baseline_with_batchnorm_model)
from preprocess import clean_text
from encoding import create_vectorize_layer, encode_labels
from custom_word_embedding import customised_embedding


class Pipeline:
    def __init__(self, params):
        self.params = params
        self.dataset_url = params.dataset_url
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
        self.dataset = pd.read_csv(self.dataset_url)

        # Sample dataset only when it is given by command line
        if self.params.nof_samples is not None:
            self.dataset = self.dataset.sample(self.params.nof_samples,
                                               random_state=self.params.seed)

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
        # Get the longest title length
        length_long_sentence = (
            self.dataset["cleaned_title"]
                .apply(lambda s: s.split())
                .apply(lambda s: len(s))
                .max()
        )
        # Set frequencies as the vectorize method
        output_mode = "int"

        # Create vectorize layer
        self.vectorize_layer = create_vectorize_layer(length_long_sentence,
                                                      output_mode)

        # Fit vectorize layer but only with training data
        self.vectorize_layer.adapt(self.x_train.values)

        # Save vocab size
        self.vocab_size = len(self.vectorize_layer.get_vocabulary())

        # Encode training and testing labels
        self.y_train, self.y_test = encode_labels(self.y_train, self.y_test)

        # Choose between pretrained or custom embeddings
        if self.params.embedding_type == "pretrained":
            self.embedding_matrix, nof_hits, nof_misses = load_embedding(
                self.params.embedding_url, self.vocab,
                self.params.embedding_dim)

            # Track the number of hits, i.e, vocab words that occurred in the
            # pretrained embedding
            mlflow.log_metric("nof hits / pretrained emb", nof_hits)
            mlflow.log_metric("nof misses / pretrained emb", nof_misses)
        elif self.params.embedding_type == "custom":
            # Train custom embedding
            self.embedding_matrix = customised_embedding(
                self.vectorize_layer(self.x_train).numpy(), self.vocab_size,
                self.params.seed, self.params.embedding_dim)

        # Create embedding layer
        self.embedding_layer = create_embedding_layer(
            vocab_size=self.vocab_size,
            embedding_dim=self.params.embedding_dim,
            embedding_matrix=self.embedding_matrix,
            input_length=length_long_sentence)

    def select_model(self):
        # Choose previously defined models
        if self.params.model == "base":
            self.model = baseline_model(
                self.vectorize_layer,
                self.embedding_layer,
                nof_classes=len(np.unique(self.y_train)))
        elif self.params.model == "base_wd":
            self.model = baseline_with_dropout_model(
                self.vectorize_layer,
                self.embedding_layer,
                nof_classes=len(np.unique(self.y_train)))
        elif self.params.model == "base_wbn":
            self.model = baseline_with_batchnorm_model(
                self.vectorize_layer,
                self.embedding_layer,
                nof_classes=len(np.unique(self.y_train)))

    def k_fold_cross_validation(self):
        # Log parameters
        mlflow.log_param("seed", self.params.seed)
        mlflow.log_param("kfolds", self.params.kfolds)
        mlflow.log_param("epochs", self.params.epochs)
        mlflow.log_param("batch size", self.params.batch_size)

        # Save initial weights
        initial_weigths = self.model.get_weights()

        # Init validation metric recorders
        val_accuracy = []
        val_loss = []
        val_balanced_accuracy = []

        # Split on training to get cross-validation sets
        skf = StratifiedKFold(n_splits=self.params.kfolds, shuffle=False)
        for fold_id, (train_indices, val_indices) in enumerate(
                skf.split(self.x_train, self.y_train)):
            # Fit the model and get the history of improvement
            history = self.model.fit(self.x_train.iloc[train_indices],
                                     self.y_train[train_indices],
                                     batch_size=self.params.batch_size,
                                     epochs=self.params.epochs,
                                     verbose=1)

            # Evaluate in validation data
            loss, accuracy = self.model.evaluate(
                self.x_train.iloc[val_indices],
                self.y_train[val_indices],
                batch_size=self.params.batch_size,
                verbose=1)

            # Predict validation labels
            y_pred = np.argmax(self.model.predict(
                self.x_train.iloc[val_indices]),
                               axis=-1)

            # Calculate balanced accuracy
            balanced_accuracy = balanced_accuracy_score(
                self.y_train[val_indices], y_pred)

            # Record accuracy, loss, and balanced accuracy
            val_accuracy.append(accuracy)
            val_loss.append(loss)
            val_balanced_accuracy.append(balanced_accuracy)

            # Save validation accuracy and loss curves so they can be displayed
            # by mlflow
            for epoch in range(self.params.epochs):
                mlflow.log_metric(f"fold {fold_id} accuracy curve",
                                  history.history["accuracy"][epoch],
                                  step=epoch)
                mlflow.log_metric(f"fold {fold_id} loss curve",
                                  history.history["loss"][epoch],
                                  step=epoch)

            # Reset model for next iteration since fit method doesn't overwrite
            # previous training iterations
            self.model.set_weights(initial_weigths)

        # Record mean accuracy and loss
        mlflow.log_metric("mean val accuracy", np.mean(val_accuracy))
        mlflow.log_metric("mean val loss", np.mean(val_loss))
        mlflow.log_metric("mean val balanced accuracy",
                          np.mean(balanced_accuracy))

    def evaluate_model(self):
        # Train model again but this time using all training data
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=self.params.batch_size,
                                 epochs=self.params.epochs,
                                 verbose=1)

        # Evaluate in testing data
        test_loss, test_accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            batch_size=self.params.batch_size,
            verbose=1)

        # Predict test labels
        y_pred = np.argmax(self.model.predict(self.x_test), axis=-1)

        # Calculate balanced accuracy
        test_balanced_accuracy = balanced_accuracy_score(self.y_test, y_pred)

        # Save testing accuracy and loss curves so they can be displayed
        # by mlflow
        for epoch in range(self.params.epochs):
            mlflow.log_metric(f"test accuracy curve",
                              history.history["accuracy"][epoch],
                              step=epoch)
            mlflow.log_metric(f"test loss curve",
                              history.history["loss"][epoch],
                              step=epoch)

        # Record test accuracy and loss.
        mlflow.log_metric("test accuracy", test_accuracy)
        mlflow.log_metric("test loss", test_loss)
        mlflow.log_metric("test balanced accuracy", test_balanced_accuracy)

    def save_predictions(self):
        y_pred = np.argmax(self.model.predict(self.x_test), axis=-1)
        predictions = pd.DataFrame(data={
            "y_pred": y_pred,
            "y_test": self.y_test
        })
        predictions.to_csv(f"{self.params.model}_predictions.csv", index=False)

    def run(self):
        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.encode_data()
        self.select_model()
        self.k_fold_cross_validation()
        self.evaluate_model()
        self.save_predictions()
