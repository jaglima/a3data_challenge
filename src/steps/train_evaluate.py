import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def train_and_evaluate(train_x, train_y, test_x, test_y):
 
    ################### MLFLOW ###############################
    mlflow.set_experiment("Iris RandomForest Experiment")

    # PARAMS 
    # FIXME make it more flexible
    MAX_DEPTH = 3
    N_ESTIMATOR = 50
    
    with mlflow.start_run():
        rf_classifier = RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATOR, random_state=42)
        rf_classifier.fit(train_x, train_y)

        y_pred = rf_classifier.predict(test_x)

        # Calculate metrics
        accuracy = accuracy_score(test_y, y_pred)
        precision = precision_score(test_y, y_pred, average='macro')
        recall = recall_score(test_y, y_pred, average='macro')
        f1score = f1_score(test_y, y_pred, average='macro')

        # Log parameters, metrics, and the model
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("n_estimators", N_ESTIMATOR)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1score)
        # Specify metric conditions
       
        mlflow.sklearn.log_model(rf_classifier, "random_forest_model")


        exp = mlflow.search_runs(
            filter_string=f"metrics.f1_score >= 0.9",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        # Get the ID of the registered model
        model_id = exp.iloc[0].run_id

        # Save the registered model as a pickle file
        mlflow.sklearn.save_model(f"runs:/{model_id}/random_forest_model", f"models/{model_id}")
               
        print(f"Run with max_depth={MAX_DEPTH}, n_estimators={N_ESTIMATOR} logged with accuracy={accuracy}")

        return model_id