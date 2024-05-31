import mlflow


def evaluate():
    threshold = 0.9
    exp = mlflow.search_runs(
        filter_string=f"metrics.f1_score >= {threshold}",
        order_by=["start_time DESC"],
        max_results=1,
    )

    # Get the ID of the registered model
    model_id = exp.iloc[0].run_id

    # Save the registered model as a pickle file
    mlflow.sklearn.save_model(
        f"runs:/{model_id}/random_forest_model", f"models/{model_id}"
    )

    return model_id
