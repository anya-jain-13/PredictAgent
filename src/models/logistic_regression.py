# Importing pandas for loading the engineered training and validation datasets
import pandas as pd

# Importing joblib for saving the trained Logistic Regression model
import joblib

# Importing Path for creating and handling project file paths
from pathlib import Path

# Importing LogisticRegression for creating the baseline classification model
from sklearn.linear_model import LogisticRegression

# Importing evaluation metrics for measuring classification performance
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# Finding the main PredictAgent project folder from the location of this Python file
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating paths to the engineered training and validation datasets
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation_features.csv"


# Creating the path where the trained Logistic Regression model will be saved
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.pkl"


# Loading the engineered training dataset
train_df = pd.read_csv(TRAIN_PATH)

# Loading the engineered validation dataset
validation_df = pd.read_csv(VALIDATION_PATH)


# Creating the list of model input columns by excluding the timestamp and prediction target
feature_columns = [
    column
    for column in train_df.columns
    if column not in ["window_end_time", "target"]
]


# Selecting the 51 engineered input features from the training dataset
X_train = train_df[feature_columns]

# Selecting the failure prediction labels from the training dataset
y_train = train_df["target"]


# Selecting the same 51 engineered input features from the validation dataset
X_validation = validation_df[feature_columns]

# Selecting the failure prediction labels from the validation dataset
y_validation = validation_df["target"]


# Creating the Logistic Regression model with automatic class balancing
model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)


# Training the Logistic Regression model using only the training dataset
model.fit(X_train, y_train)


# Calculating failure probabilities for every validation window
validation_probabilities = model.predict_proba(X_validation)[:, 1]


# Converting probabilities into initial predictions using the default 0.50 threshold
validation_predictions = (
    validation_probabilities >= 0.50
).astype(int)


# Calculating the confusion matrix for the validation predictions
confusion = confusion_matrix(
    y_validation,
    validation_predictions
)


# Calculating ROC-AUC using the predicted probabilities
roc_auc = roc_auc_score(
    y_validation,
    validation_probabilities
)


# Calculating PR-AUC using the predicted probabilities
pr_auc = average_precision_score(
    y_validation,
    validation_probabilities
)


# Displaying the validation confusion matrix
print("\nValidation Confusion Matrix:")
print(confusion)


# Displaying precision, recall, F1-score, and support
print("\nValidation Classification Report:")
print(
    classification_report(
        y_validation,
        validation_predictions,
        digits=4,
        zero_division=0
    )
)


# Displaying the ROC-AUC score
print("Validation ROC-AUC:")
print(roc_auc)


# Displaying the Precision-Recall AUC score
print("\nValidation PR-AUC:")
print(pr_auc)


# Saving the trained Logistic Regression model for later comparison and deployment
joblib.dump(model, MODEL_PATH)


# Displaying the location of the saved model
print("\nLogistic Regression model saved to:")
print(MODEL_PATH)