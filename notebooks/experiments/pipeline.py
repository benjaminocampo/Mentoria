# Data
import numpy as np
import pandas as pd
# MLflowf
import mlflow
# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
# Utilities
from preprocess import clean_text
from encoding import encode_labels


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

        self.dataset = self.dataset.assign(
            encoded_category=encode_labels(self.dataset["category"]))

        # Set titles and categories as features and labels
        self.x = self.dataset["cleaned_title"]
        self.y = self.dataset["encoded_category"]

    def split_data(self):
        # Split dataset into training and testing instances
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=self.params.test_size,
            random_state=self.params.seed)

    def k_fold_cross_validation(self, model):
        # Log parameters
        mlflow.log_param("seed", self.params.seed)
        mlflow.log_param("kfolds", self.params.kfolds)
        mlflow.log_param("epochs", self.params.epochs)
        mlflow.log_param("batch size", self.params.batch_size)

        # Save initial weights
        initial_weigths = model.get_weights()

        # Init validation metric recorders
        stats = pd.DataFrame(columns=[
            "val_acc",
            "rel_acc",
            "unrel_acc",
            "val_loss",
            "rel_loss",
            "unrel_loss",
            "val_bcd_acc",
            "rel_bcd_acc",
            "unrel_bcd_acc"
        ])
        # Split on training to get cross-validation sets
        skf = StratifiedKFold(n_splits=self.params.kfolds, shuffle=False)
        for fold_id, (train_indices, val_indices) in enumerate(
                skf.split(self.x_train, self.y_train)):
            # Fit the model and get the history of improvement
            history = model.fit(self.x_train.values[train_indices],
                                self.y_train.values[train_indices],
                                batch_size=self.params.batch_size,
                                epochs=self.params.epochs,
                                verbose=1)

            # stats.loc[:, ["val_acc", "rel_acc", "unrel_acc"]] = self.calculate_metrics(
            #     self.x_train.iloc[val_indices], self.y_train[val_indices])
            #
            # x, y = (
            #     self.dataset
            #     .iloc[val_indices]
            #     .loc[self.dataset.label_quality == "reliable", ["cleaned_title", "category"]]
            #     .T
            #     .values
            # )
            # 
            # stats.loc[:, ["val_loss", "rel_loss", "unrel_loss"
            #               ]] = self.calculate_metrics(x, y)
            # 
            # x, y = (
            #     self.dataset
            #     .iloc[val_indices]
            #     .loc[self.dataset.label_quality == "unreliable", ["cleaned_title", "category"]]
            #     .T
            #     .values
            # )
            # 
            # stats.loc[:, ["val_bcd_acc", "rel_bcd_acc", "unrel_bcd_acc"
            #               ]] = self.calculate_metrics(x, y)


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
            model.set_weights(initial_weigths)

        # Record mean accuracy and loss
        mlflow.log_metric("mean val accuracy", np.mean(stats["val_accuracy"]))
        mlflow.log_metric("mean val loss", np.mean(stats["val_loss"]))
        mlflow.log_metric("mean val balanced accuracy",
                          np.mean(stats["balanced_accuracy"]))

        # mlflow.log_metric("mean val accuracy", np.mean(stats["rel_acc"]))
        # mlflow.log_metric("mean val loss", np.mean(stats["rel_loss"]))
        # mlflow.log_metric("mean val balanced accuracy",
        #                   np.mean(stats["unrel_loss"]))
        #
        # mlflow.log_metric("mean val accuracy", np.mean(stats["unrel_acc"]))
        # mlflow.log_metric("mean val loss", np.mean(stats["unrel_loss"]))
        # mlflow.log_metric("mean val balanced accuracy",
        #                   np.mean(stats["unrel_bcd_acc"]))

    def calculate_metrics(self, x, y):
        # Evaluate in validation data
        loss, accuracy = self.model.evaluate(x,
                                             y,
                                             batch_size=self.params.batch_size,
                                             verbose=1)
        # Predict validation labels
        y_pred = np.argmax(self.model.predict(x), axis=-1)
        # Calculate balanced accuracy
        balanced_accuracy = balanced_accuracy_score(y, y_pred)

        return loss, accuracy, balanced_accuracy


    def evaluate_model(self):
        # Train model again but this time using all training data
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=self.params.batch_size,
                                 epochs=self.params.epochs,
                                 verbose=1)

        test_accuracy, test_loss, test_balanced_accuracy = self.calculate_metrics(
            self.x_test, self.y_test)
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
        self.k_fold_cross_validation()
        self.evaluate_model()
        self.save_predictions()
