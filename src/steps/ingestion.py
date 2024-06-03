import pandas as pd


def download_iris_dataset():
    # URL para o conjunto de dados Iris com cabeçalho e rótulos como texto
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    df = pd.read_csv("data/iris.data", header=None, names=column_names)

    return df
