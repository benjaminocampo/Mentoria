def calculate_metrics(self, x, y):
    # Evaluate in validation data
    loss, accuracy = model.evaluate(x,
                                         y,
                                         batch_size=params.batch_size,
                                         verbose=1)
    # Predict validation labels
    y_pred = np.argmax(model.predict(x), axis=-1)
    # Calculate balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    return loss, accuracy, balanced_accuracy

def mflow_logs():
    # Log parameters
    mlflow.log_param("seed", params.seed)
    mlflow.log_param("kfolds", params.kfolds)
    mlflow.log_param("epochs", params.epochs)
    mlflow.log_param("batch size", params.batch_size)

    for epoch in range(params.epochs):
        mlflow.log_metric(f"fold {fold_id} accuracy curve",
                          history.history["accuracy"][epoch],
                          step=epoch)
        mlflow.log_metric(f"fold {fold_id} loss curve",
                          history.history["loss"][epoch],
                          step=epoch)
    
        # Record mean accuracy and loss
    mlflow.log_metric("mean val accuracy", np.mean(stats["val_accuracy"]))
    mlflow.log_metric("mean val loss", np.mean(stats["val_loss"]))
    mlflow.log_metric("mean val balanced accuracy",
                      np.mean(stats["balanced_accuracy"]))

    mlflow.log_metric("mean val accuracy", np.mean(stats["rel_acc"]))
    mlflow.log_metric("mean val loss", np.mean(stats["rel_loss"]))
    mlflow.log_metric("mean val balanced accuracy",
                      np.mean(stats["unrel_loss"]))

    mlflow.log_metric("mean val accuracy", np.mean(stats["unrel_acc"]))
    mlflow.log_metric("mean val loss", np.mean(stats["unrel_loss"]))
    mlflow.log_metric("mean val balanced accuracy",
                      np.mean(stats["unrel_bcd_acc"]))

    for epoch in range(params.epochs):
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