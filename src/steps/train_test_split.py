from sklearn.model_selection import train_test_split


def split_data(df, test_ratio, random_state):
    train, test = train_test_split(df, test_size=test_ratio, random_state=random_state)

    return train, test


def get_feat_and_target(df, target):
    """
    Get features and target variables seperately from given dataframe and target
    input: dataframe and target column
    output: two dataframes for x and y
    """
    x = df.drop(target, axis=1)
    y = df[[target]]
    y = y.values.ravel()
    return x, y
