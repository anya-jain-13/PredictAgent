# Importing pandas for loading, splitting, and saving the labelled MetroPT dataset
import pandas as pd

# Importing Path for creating and handling the project file paths
from pathlib import Path


# Finding the main PredictAgent project folder from the location of this Python file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Creating the path to the labelled MetroPT dataset
LABELLED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "metropt_labelled.csv"


# Loading the labelled MetroPT dataset into a pandas DataFrame
df = pd.read_csv(LABELLED_DATA_PATH)

# Converting the timestamp column from text into datetime values for chronological splitting
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sorting all readings chronologically before creating the train, validation, and test datasets
df = df.sort_values("timestamp").reset_index(drop=True)


# Displaying the complete time range available in the MetroPT dataset
print("Complete dataset time range:")
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())


# Displaying the documented failure periods present in the labelled dataset
failure_periods = df[df["failure_active"] == 1]

print("\nActive failure data range:")
print("First failure reading:", failure_periods["timestamp"].min())
print("Last failure reading:", failure_periods["timestamp"].max())


# Displaying the number of positive prediction labels available in the complete dataset
print("\nTotal next-6-hour positive readings:")
print(df["failure_next_6h"].sum())

# Creating the timestamp where the validation dataset begins
validation_start = pd.Timestamp("2020-06-01 00:00:00")

# Creating the timestamp where the test dataset begins
test_start = pd.Timestamp("2020-07-01 00:00:00")


# Selecting all readings before June for training the prediction models
train_df = df[df["timestamp"] < validation_start].copy()

# Selecting June readings for validating models and selecting model settings
validation_df = df[
    (df["timestamp"] >= validation_start)
    & (df["timestamp"] < test_start)
].copy()

# Selecting July and later readings for the final evaluation of the trained model
test_df = df[df["timestamp"] >= test_start].copy()


# Creating paths for saving the three chronological datasets
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"


# Saving the training dataset without creating an additional CSV index column
train_df.to_csv(TRAIN_PATH, index=False)

# Saving the validation dataset without creating an additional CSV index column
validation_df.to_csv(VALIDATION_PATH, index=False)

# Saving the test dataset without creating an additional CSV index column
test_df.to_csv(TEST_PATH, index=False)


# Displaying the number of readings present in each chronological dataset
print("\nDataset sizes:")
print("Train:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))


# Displaying the time range covered by the training dataset
print("\nTraining time range:")
print(train_df["timestamp"].min(), "to", train_df["timestamp"].max())

# Displaying the time range covered by the validation dataset
print("\nValidation time range:")
print(validation_df["timestamp"].min(), "to", validation_df["timestamp"].max())

# Displaying the time range covered by the test dataset
print("\nTest time range:")
print(test_df["timestamp"].min(), "to", test_df["timestamp"].max())


# Displaying the number of next-6-hour positive labels available for model training
print("\nTraining positive labels:")
print(train_df["failure_next_6h"].sum())

# Displaying the number of next-6-hour positive labels available for model validation
print("\nValidation positive labels:")
print(validation_df["failure_next_6h"].sum())

# Displaying the number of next-6-hour positive labels reserved for final model testing
print("\nTest positive labels:")
print(test_df["failure_next_6h"].sum())