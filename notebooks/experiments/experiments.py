# %%
# !pip install mlflow
# !pip install keras
# !pip install gensim --upgrade
# %%
import mlflow
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform, randint

from models import ff_model, lstm_model

import warnings
warnings.filterwarnings("module", category=DeprecationWarning)
# %%
URL = "https://www.famaf.unc.edu.ar/~nocampo043/ML_2019_challenge_dataset_preprocessed.csv"
NOF_SAMPLES = 20000
SEED = 0

df = pd.read_csv(URL)
df.sample(n=NOF_SAMPLES, random_state=SEED)
# %%
x = df
y = df["encoded_category"]
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=SEED)

x_test_unrel = x_test[x_test["label_quality"] == "unreliable"]
y_test_unrel = x_test_unrel["encoded_category"]

x_test_rel = x_test[x_test["label_quality"] == "reliable"]
y_test_rel = x_test_rel["encoded_category"]
# %%
length_long_sentence = (df["cleaned_title"].apply(lambda s: s.split()).apply(
    lambda s: len(s)).max())
# %%
mlflow.set_experiment("LSTM vs Feed Forward")

with mlflow.start_run():
    build_ff = lambda units, dropout, lr, embedding_dim: ff_model(
        x_train["cleaned_title"], length_long_sentence, units, dropout, lr, embedding_dim)

    build_lstm = lambda units, dropout, lr, embedding_dim: lstm_model(
        x_train["cleaned_title"], length_long_sentence, units, dropout, lr, embedding_dim)

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
        (ff, dist),
        (lstm, dist)
    ]

    for hyper_model, dist in searches:
        mlflow.log_params(dist)

        clf = RandomizedSearchCV(estimator=hyper_model,
                                 param_distributions=dist,
                                 cv=5,
                                 scoring="balanced_accuracy",
                                 verbose=3)

        clf.fit(x_train["cleaned_title"], y_train)

        y_pred = clf.best_estimator_.predict(x_test["cleaned_title"])
        y_pred_rel = clf.best_estimator_.predict(x_test_rel["cleaned_title"])
        y_pred_unrel = clf.best_estimator_.predict(x_test_unrel["cleaned_title"])

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
        predictions.to_csv(f"predictions.csv", index=False)
# %%
