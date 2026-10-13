# Importing pandas for loading and processing the train, validation, and test datasets
import pandas as pd

# Importing joblib for saving the fitted scaler so the same scaler can be used during future predictions
import joblib

# Importing StandardScaler for standardizing the continuous analogue sensor measurements
from sklearn.preprocessing import StandardScaler

# Importing Path for creating and handling project file paths
from pathlib import Path


# Finding the main PredictAgent project folder from the location of this Python file
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating paths to the chronological train, validation, and test datasets
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


# Creating paths where the scaled datasets will be saved
SCALED_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train_scaled.csv"
SCALED_VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation_scaled.csv"
SCALED_TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_scaled.csv"


# Creating the path where the fitted scaler will be saved for future model inference
SCALER_PATH = PROJECT_ROOT / "models" / "sensor_scaler.pkl"


# Loading the training dataset
train_df = pd.read_csv(TRAIN_PATH)

# Loading the validation dataset
validation_df = pd.read_csv(VALIDATION_PATH)

# Loading the test dataset
test_df = pd.read_csv(TEST_PATH)


# Listing the continuous analogue sensor columns that require standardization
analogue_columns = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current"
]


# Creating a StandardScaler for standardizing the analogue sensor measurements
scaler = StandardScaler()


# Learning the mean and standard deviation of every analogue sensor using only the training data
scaler.fit(train_df[analogue_columns])


# Standardizing the training analogue sensor values using the statistics learned from the training data
train_df[analogue_columns] = scaler.transform(train_df[analogue_columns])

# Standardizing the validation analogue sensor values using the same training-data statistics
validation_df[analogue_columns] = scaler.transform(
    validation_df[analogue_columns]
)

# Standardizing the test analogue sensor values using the same training-data statistics
test_df[analogue_columns] = scaler.transform(
    test_df[analogue_columns]
)


# Saving the fitted scaler so future real-time sensor readings can be transformed in exactly the same way
joblib.dump(scaler, SCALER_PATH)


# Saving the scaled training dataset
train_df.to_csv(SCALED_TRAIN_PATH, index=False)

# Saving the scaled validation dataset
validation_df.to_csv(SCALED_VALIDATION_PATH, index=False)

# Saving the scaled test dataset
test_df.to_csv(SCALED_TEST_PATH, index=False)


# Displaying the mean of each scaled analogue sensor in the training dataset
print("Training sensor means after scaling:")
print(train_df[analogue_columns].mean())


# Displaying the standard deviation of each scaled analogue sensor in the training dataset
print("\nTraining sensor standard deviations after scaling:")
print(train_df[analogue_columns].std())


# Displaying the locations of the newly created scaled datasets
print("\nScaled datasets saved successfully:")
print(SCALED_TRAIN_PATH)
print(SCALED_VALIDATION_PATH)
print(SCALED_TEST_PATH)


# Displaying the location of the fitted scaler
print("\nScaler saved to:")
print(SCALER_PATH)