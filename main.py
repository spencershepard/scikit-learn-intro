import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Embarked", "Fare"], inplace=True)
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    # if age is missing, drop the row
    df.dropna(subset=["Age"], inplace=True)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)
    df.dropna(subset=["Survived"], inplace=True)
    return df

def tune_model(X_train, y_train):
    param_grid = {
        'n_neighbors': range(1, 20),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion matrix:")
    print(cm)
    return cm


def plot_model(matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Survived", "Survived"],
                yticklabels=["Not Survived", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    data = preprocess_data(pd.read_csv("titanic.csv"))
    print(data.info)
    print(data.isnull().sum())

    # Check for missing values
    if data.isnull().values.any():
        print("Data contains missing values. Please check the dataset.")
        exit()

    # Check for duplicate rows
    if data.duplicated().any():
        print("Data contains duplicate rows. Removing duplicates...")
        data = data.drop_duplicates()

    # Print the actual number of survivors and non-survivors
    print("Number of survivors:", data["Survived"].sum())
    print("Number of non-survivors:", len(data) - data["Survived"].sum())

    x = data.drop(columns=["Survived"])  # "the question"
    y = data["Survived"]  # "the answer"

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model = tune_model(X_train_scaled, y_train)
    matrix = evaluate_model(best_model, X_test_scaled, y_test)
    plot_model(matrix)



