# Importing pandas for loading validation features
import pandas as pd

# Importing NumPy for testing different probability thresholds
import numpy as np

# Importing joblib for loading the trained Logistic Regression model
import joblib

# Importing Path for creating project file paths
from pathlib import Path

# Importing classification metrics for comparing thresholds
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Finding the main PredictAgent project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating the path to the validation feature dataset
VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_features.csv"
)


# Creating the path to the trained Logistic Regression model
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression.pkl"
)


# Loading the validation feature dataset
validation_data = pd.read_csv(
    VALIDATION_PATH
)


# Selecting the target column
y_validation = validation_data["target"]


# Removing metadata and target columns from the model inputs
X_validation = validation_data.drop(
    columns=[
        "window_end_time",
        "target"
    ]
)


# Loading the trained Logistic Regression model
model = joblib.load(
    MODEL_PATH
)


# Generating failure probabilities for the validation windows
probabilities = model.predict_proba(
    X_validation
)[:, 1]


# Creating thresholds from 0.01 to 0.99
thresholds = np.arange(
    0.01,
    1.00,
    0.01
)


# Creating an empty list for storing threshold results
results = []


# Evaluating every probability threshold
for threshold in thresholds:

    # Converting probabilities into binary predictions
    predictions = (
        probabilities >= threshold
    ).astype(int)


    # Calculating failure precision
    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0
    )


    # Calculating failure recall
    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0
    )


    # Calculating the failure F1 score
    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0
    )


    # Storing the results for the current threshold
    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    )


# Converting all threshold results into a DataFrame
results_df = pd.DataFrame(
    results
)


# Finding the threshold with the highest F1 score
best_row = results_df.loc[
    results_df["f1_score"].idxmax()
]


# Extracting the best threshold
best_threshold = float(
    best_row["threshold"]
)


# Creating predictions using the selected threshold
best_predictions = (
    probabilities >= best_threshold
).astype(int)


# Calculating the confusion matrix at the selected threshold
best_confusion_matrix = confusion_matrix(
    y_validation,
    best_predictions
)


# Displaying the best threshold
print("\nBest validation threshold:")
print(best_threshold)


# Displaying precision at the selected threshold
print("\nFailure precision:")
print(best_row["precision"])


# Displaying recall at the selected threshold
print("\nFailure recall:")
print(best_row["recall"])


# Displaying F1 score at the selected threshold
print("\nFailure F1 score:")
print(best_row["f1_score"])


# Displaying the confusion matrix at the selected threshold
print("\nConfusion Matrix:")
print(best_confusion_matrix)


# Displaying the ten thresholds with the highest F1 scores
print("\nTop 10 thresholds by F1 score:")

print(
    results_df
    .sort_values(
        "f1_score",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)