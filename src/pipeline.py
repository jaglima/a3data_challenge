from steps.ingestion import load_data
from steps.transformers import preprocess_iris_dataframe
from steps.train_test_split import split_data, get_feat_and_target
from steps.train import train_and_evaluate
from steps.evaluate import evaluate


def run():
    # It is a imitation of orchestrator
    # Load Data
    df = load_data()

    # Features and target transformations
    df = preprocess_iris_dataframe(df, "target")

    # Train and test split
    train, test = split_data(df, test_ratio=0.7, random_state=42)
    train_x, train_y = get_feat_and_target(train, target="target")
    test_x, test_y = get_feat_and_target(test, target="target")

    # train and evaluation
    train_and_evaluate(train_x, train_y, test_x, test_y)

    # evaluate model
    evaluate()


if __name__ == "__main__":
    run()
