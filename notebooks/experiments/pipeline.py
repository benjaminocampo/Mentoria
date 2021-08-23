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
from encoding import encode_labels, tokenize_features
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
        self.train_label_quality = None
        self.test_label_quality = None
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
        self.x = self.dataset[["cleaned_title","label_quality"]]
        self.y = self.dataset["category"]


    def split_data(self):
        # Split dataset into training and testing instances
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=self.params.test_size,
            random_state=self.params.seed)

        self.train_label_quality = self.x_train["label_quality"]
        self.test_label_quality = self.x_test["label_quality"]
        
        self.x_train = self.x_train["cleaned_title"]
        self.x_test = self.x_test["cleaned_title"]
        


    def encode_data(self):
        # Encode training and testing titles separately
        self.x_train, self.x_test, self.vocab = tokenize_features(
            self.x_train, self.x_test)

        # Encode training and testing labels
        self.y_train, self.y_test = encode_labels(self.y_train, self.y_test)


        if self.params.embedding_type == "pretrained":
            # Load pre-trained embeddings
            self.embeddings, nof_hits, nof_misses = load_embedding(
                self.params.embedding_url, self.vocab, self.params.embedding_dim)
            # Track the number of words of the vocab that appeared in the
            # pre-trained embedding
            mlflow.log_metric("nof hits / pretrained emb", nof_hits)
            mlflow.log_metric("nof misses / pretrained emb", nof_misses)
        elif self.params.embedding_type == "custom":
            # Apply custom embedding
            self.embeddings = customised_embedding(self.x_train, self.vocab,
                                                   self.params.seed,
                                                   self.params.embedding_dim)

    def select_model(self):
        # Get max sentence length
        _, max_sentence_length = self.x_train.shape

        # Create embedding layer
        embedding_layer = create_embedding_layer(
            vocab_size=len(self.vocab) + 1,
            embedding_dim=self.params.embedding_dim,
            embedding_matrix=self.embeddings,
            input_length=max_sentence_length)

        # Choose model
        # TODO: Here we should choose over a variaty of models. They could also
        # be implemented using
        if self.params.model == "base":
            self.model = baseline_model(embedding_layer=embedding_layer,
                                        nof_classes=len(np.unique(self.y_train)))
        elif self.params.model == "base_wd":
            self.model = baseline_with_dropout_model(embedding_layer=embedding_layer,
                                        nof_classes=len(np.unique(self.y_train)))
        elif self.params.model == "base_wbn":
            self.model = baseline_with_batchnorm_model(embedding_layer=embedding_layer,
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
        val_label_quality_reliable = []
        val_label_quality_unreliable = []

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

            val_reliable_indices = self.train_label_quality.loc[self.train_label_quality["label_quality"] == "reliable", val_indices].index

            _,accuracy_label_quality_reliable = self.model.evaluate(
                self.x_train[val_reliable_indices],
                self.y_train[val_reliable_indices],
                batch_size=self.params.batch_size,
                verbose=1)

            val_unreliable_indices = self.train_label_quality.loc[self.train_label_quality["label_quality"] == "unreliable", val_indices].index

            _,accuracy_label_quality_unreliable = self.model.evaluate(
                self.x_train[val_unreliable_indices],
                self.y_train[val_unreliable_indices],
                batch_size=self.params.batch_size,
                verbose=1)

            # Predict validation labels
            y_pred = np.argmax(self.model.predict(self.x_train[val_indices]),
                               axis=-1)

            # Calculate balanced accuracy
            balanced_accuracy = balanced_accuracy_score(
                self.y_train[val_indices], y_pred)

            # Record accuracy, loss, and balanced accuracy
            val_accuracy.append(accuracy)
            val_loss.append(loss)
            val_balanced_accuracy.append(balanced_accuracy)
            val_label_quality_reliable.append(accuracy_label_quality_reliable)
            val_label_quality_unreliable.append(accuracy_label_quality_unreliable)




            # Save validation accuracy and loss curves so they can be displayed
            # by mlflow
            for epoch in range(self.params.epochs):
                mlflow.log_metric(f"fold {fold_id} accuracy curve",history.history["accuracy"][epoch],step=epoch)
                mlflow.log_metric(f"fold {fold_id} loss curve",history.history["loss"][epoch],step=epoch)

            # Reset model for next iteration since fit method doesn't overwrite
            # previous training iterations
            self.model.set_weights(initial_weigths)

        # Record mean accuracy and loss
        mlflow.log_metric("mean val accuracy", np.mean(val_accuracy))
        mlflow.log_metric("mean val loss", np.mean(val_loss))
        mlflow.log_metric("mean val balanced accuracy", np.mean(balanced_accuracy))
        mlflow.log_metric("mean val reliable accuracy", np.mean(val_label_quality_reliable))
        mlflow.log_metric("mean val unreliable accuracy", np.mean(val_label_quality_unreliable))


    def evaluate_model(self):
        # Train model again but this time using all training data
        history = self.model.fit(self.x_train,
                                 self.y_train,
                                 batch_size=self.params.batch_size,
                                 epochs=self.params.epochs,
                                 verbose=1)

        # Evaluate in testing data
        test_loss, test_accuracy = self.model.evaluate(self.x_test,
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
            mlflow.log_metric(f"test accuracy curve",history.history["accuracy"][epoch],step=epoch)
            mlflow.log_metric(f"test loss curve",history.history["loss"][epoch],step=epoch)


        test_reliable_indices = self.test_label_quality.loc[self.test_label_quality["label_quality"] == "reliable"].index

        _, accuracy_test_label_quality_reliable = self.model.evaluate(
            self.x_train[test_reliable_indices],
            self.y_train[test_reliable_indices],
            batch_size=self.params.batch_size,
            verbose=1)

        test_unreliable_indices = self.test_label_quality.loc[self.test_label_quality["label_quality"] == "unreliable"].index

        _, accuracy_test_label_quality_unreliable = self.model.evaluate(
            self.x_train[test_unreliable_indices],
            self.y_train[test_unreliable_indices],
            batch_size=self.params.batch_size,
            verbose=1)



        # Record test accuracy and loss.
        mlflow.log_metric("test accuracy", test_accuracy)
        mlflow.log_metric("test loss", test_loss)
        mlflow.log_metric("test balanced accuracy", test_balanced_accuracy)
        mlflow.log_metric("test reliable accuracy", accuracy_test_label_quality_reliable)
        mlflow.log_metric("test unreliable accuracy",accuracy_test_label_quality_unreliable)


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
