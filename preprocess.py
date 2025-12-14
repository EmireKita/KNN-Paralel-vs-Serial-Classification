import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess(path, test_size=0.2):
    df = pd.read_csv(path)

    # pisahkan fitur dan label
    X = df.drop(columns=['label']).values
    y = df['label'].values

    # normalisasi fitur
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test
