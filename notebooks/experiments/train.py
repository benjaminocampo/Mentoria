from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from keras_tuner import Objective
from keras_tuner.tuners import SklearnTuner, Hyperband
from keras_tuner.oracles import BayesianOptimizationOracle
import sklearn.pipeline


def k_fold_cross_validation(x_train,
                            y_train,
                            hyper_model,
                            kfolds=5):

    tuner = Hyperband(hyper_model,
                      objective='val_accuracy',
                      max_epochs=10,
                      factor=3,
                      overwrite=True)
    
    tuner.search(x_train, y_train, epochs=5, validation_split=0.2)
    best_model = tuner.get_best_models()[0]

    # tuner = SklearnTuner(oracle=BayesianOptimizationOracle(objective=Objective(
    #     'score', 'max'),
    #                                                        max_trials=10),
    #                      hypermodel=hyper_model,
    #                      scoring=balanced_accuracy_score,
    #                      cv=StratifiedKFold(kfolds))
    # tuner.search(x_train.to_numpy(), y_train.to_numpy())
    # best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

# stats.loc[:, ["val_acc", "rel_acc", "unrel_acc"]] = calculate_metrics(
#     x_train.iloc[val_indices], y_train[val_indices])
#
# x, y = (
#     dataset
#     .iloc[val_indices]
#     .loc[dataset.label_quality == "reliable", ["cleaned_title", "category"]]
#     .T
#     .values
# )
#
# stats.loc[:, ["val_loss", "rel_loss", "unrel_loss"
#               ]] = calculate_metrics(x, y)
#
# x, y = (
#     dataset
#     .iloc[val_indices]
#     .loc[dataset.label_quality == "unreliable", ["cleaned_title", "category"]]
#     .T
#     .values
# )
#
# stats.loc[:, ["val_bcd_acc", "rel_bcd_acc", "unrel_bcd_acc"
#               ]] = calculate_metrics(x, y)
# Reset model for next iteration since fit method doesn't overwrite
# previous training iterations

# def evaluate_model(self):
#     # Train model again but this time using all training data
#     history = model.fit(x_train,
#                              y_train,
#                              batch_size=params.batch_size,
#                              epochs=params.epochs,
#                              verbose=1)
#     test_accuracy, test_loss, test_balanced_accuracy = calculate_metrics(
#         x_test, y_test)
# 
# 
# def save_predictions(self):
#     y_pred = np.argmax(model.predict(x_test), axis=-1)
#     predictions = pd.DataFrame(data={
#         "y_pred": y_pred,
#         "y_test": y_test
#     })
#     predictions.to_csv(f"{params.model}_predictions.csv", index=False)