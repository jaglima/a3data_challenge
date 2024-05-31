from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def preprocess_iris_dataframe(df, target_column):
    # Copy the dataframe to avoid modifying the original one
    df_preprocessed = df.copy()
    
    # Scale the feature columns
    scaler = StandardScaler()
    feature_columns = df_preprocessed.columns[df_preprocessed.columns != target_column]
    df_preprocessed[feature_columns] = scaler.fit_transform(df_preprocessed[feature_columns])
    
    # Encode the target column
    label_encoder = LabelEncoder()
    df_preprocessed[target_column] = label_encoder.fit_transform(df_preprocessed[target_column])
    
    return df_preprocessed, label_encoder