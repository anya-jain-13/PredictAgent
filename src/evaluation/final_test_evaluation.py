# Importing pandas for loading the untouched test feature dataset
import pandas as pd

# Importing joblib for loading the trained Logistic Regression model
import joblib

# Importing Path for creating project file paths
from pathlib import Path

# Importing classification metrics for final model evaluation
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# Finding the main PredictAgent project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating the path to the untouched test feature dataset
TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_features.csv"
)


# Creating the path to the selected Logistic Regression model
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression.pkl"
)


# Using the threshold selected previously on the validation dataset
SELECTED_THRESHOLD = 0.97


# Loading the untouched test feature dataset
test_data = pd.read_csv(
    TEST_PATH
)


# Selecting the true test targets
y_test = test_data["target"]


# Removing metadata and target columns from the model inputs
X_test = test_data.drop(
    columns=[
        "window_end_time",
        "target"
    ]
)


# Loading the selected Logistic Regression model
model = joblib.load(
    MODEL_PATH
)


# Generating failure probabilities for the untouched test windows
probabilities = model.predict_proba(
    X_test
)[:, 1]


# Converting probabilities into predictions using the locked threshold
predictions = (
    probabilities >= SELECTED_THRESHOLD
).astype(int)


# Calculating the final test confusion matrix
confusion = confusion_matrix(
    y_test,
    predictions
)


# Calculating failure precision
precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)


# Calculating failure recall
recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)


# Calculating the failure F1 score
f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# Calculating test ROC-AUC using probabilities
roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# Calculating test Precision-Recall AUC using probabilities
pr_auc = average_precision_score(
    y_test,
    probabilities
)


# Displaying the locked operating threshold
print("\nSelected threshold:")
print(SELECTED_THRESHOLD)


# Displaying the final test confusion matrix
print("\nTest Confusion Matrix:")
print(confusion)


# Displaying the complete classification report
print("\nTest Classification Report:")
print(
    classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )
)


# Displaying failure precision
print("Test Failure Precision:")
print(precision)


# Displaying failure recall
print("\nTest Failure Recall:")
print(recall)


# Displaying failure F1 score
print("\nTest Failure F1:")
print(f1)


# Displaying test ROC-AUC
print("\nTest ROC-AUC:")
print(roc_auc)


# Displaying test Precision-Recall AUC
print("\nTest PR-AUC:")
print(pr_auc)


# Displaying the minimum predicted probability
print("\nMinimum predicted probability:")
print(probabilities.min())


# Displaying the maximum predicted probability
print("\nMaximum predicted probability:")
print(probabilities.max())