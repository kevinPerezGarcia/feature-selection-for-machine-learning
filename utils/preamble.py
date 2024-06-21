def preamble(data_path):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(data_path)

    print(data.head())

    X = data.drop('target', axis=1)

    y = data[['target']]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0)

    print(f"The number of columns in X_train is {X.shape[1]}")

    return X_train, X_test, y_train, y_test