import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def train_and_evaluate(train_x, train_y, test_x, test_y):
    # Get the current tracking uri
    # mlflow.set_tracking_uri("http://localhost:5000")
    tracking_uri = mlflow.get_tracking_uri()
    print(f"Current tracking uri: {tracking_uri}")
    mlflow.set_experiment("Iris RandomForest Experiment")

    # PARAMS
    # FIXME make it more flexible
    MAX_DEPTH = 3
    N_ESTIMATOR = 50

    with mlflow.start_run():
        rf_classifier = RandomForestClassifier(
            max_depth=MAX_DEPTH, n_estimators=N_ESTIMATOR, random_state=42
        )
        rf_classifier.fit(train_x, train_y)

        y_pred = rf_classifier.predict(test_x)

        # Calculate metrics
        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average="macro")
        recall = recall_score(test_y, y_pred, average="macro")
        f1score = f1_score(test_y, y_pred, average="macro")

        # Log parameters, metrics, and the model
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("n_estimators", N_ESTIMATOR)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)
        # Specify metric conditions

        mlflow.sklearn.log_model(rf_classifier, "random_forest_model")

        # Log additional artifact (preprocessing object)
        mlflow.log_artifact("data/target_encoder.pkl", "random_forest_model")
        mlflow.log_artifact("data/feat_encoder.pkl", "random_forest_model")

        print(
            f"Run with max_depth={MAX_DEPTH},  n_estimators={N_ESTIMATOR} accuracy={accuracy}"
        )
