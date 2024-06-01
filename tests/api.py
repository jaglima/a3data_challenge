import mlflow.pyfunc
import mlflow.artifacts
import pandas as pd
import pickle

model_uri = "models:/champion/1"
feat_uri = mlflow.artifacts.download_artifacts(f"{model_uri}/feat_encoder.pkl")
target_uri = mlflow.artifacts.download_artifacts(f"{model_uri}/target_encoder.pkl")


# Use the loaded Label Encoder to inverse transform
cols = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
input_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=cols)
print(input_data)

# Load the Label Encoder
with open(feat_uri, "rb") as f:
    loaded_feat = pickle.load(f)
features = loaded_feat.transform(input_data)
print(features)

# Perform inference
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(input_data)

# Output predictions
print(predictions)

# Load the Label Encoder
with open(target_uri, "rb") as le:
    loaded_le = pickle.load(le)
original_label = loaded_le.inverse_transform(predictions)

print(original_label)
