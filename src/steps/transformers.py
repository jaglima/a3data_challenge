from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


def feature_transformer(df, num_cols):
    df_transformed = df.copy()
    scaler = StandardScaler()
    df_transformed[num_cols] = scaler.fit_transform(df_transformed[num_cols])

    # Save the Label Encoder
    with open("data/feat_encoder.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return df_transformed


def target_transformer(df, cat_cols):
    df_transformed = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_transformed[col] = le.fit_transform(df_transformed[col])

    # Save the Target Encoder
    with open("data/target_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return df_transformed


def preprocess_iris_dataframe(df, target_column):
    # Copy the dataframe to avoid modifying the original one
    df_preprocessed = df.copy()

    # Define feature columns, numerical columns, and categorical columns
    num_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    cat_cols = [target_column]

    # Apply feature transformation to numerical columns
    df_preprocessed = feature_transformer(df_preprocessed, num_cols)

    # Apply target transformation to categorical columns
    df_preprocessed = target_transformer(df_preprocessed, cat_cols)

    return df_preprocessed


# Example usage with Iris dataset
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Load Iris dataset
    iris = load_iris(as_frame=True)
    df_iris = iris["data"]
    df_iris["target"] = iris["target"]

    # Preprocess the Iris dataframe
    preprocessed_df = preprocess_iris_dataframe(df_iris, "target")
    print(preprocessed_df.head())
    print(df_iris.head())
