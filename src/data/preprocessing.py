# Importing pandas for loading, cleaning, and processing the MetroPT dataset
import pandas as pd

# Importing Path for creating file paths inside the PredictAgent project
from pathlib import Path


# Finding the main PredictAgent project folder from the location of this Python file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Creating the path to the original MetroPT dataset
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"

# Creating the path where the cleaned MetroPT dataset will be saved
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "metropt_cleaned.csv"


# Loading the original MetroPT dataset into a pandas DataFrame
df = pd.read_csv(RAW_DATA_PATH)

# Displaying the original dataset shape before performing preprocessing
print("Original dataset shape:")
print(df.shape)


# Removing the unnecessary CSV index column because it is not a compressor sensor
df = df.drop(columns=["Unnamed: 0"])

# Converting the timestamp column from text into datetime values for time-series processing
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sorting all sensor readings chronologically using their timestamps
df = df.sort_values(by="timestamp")

# Resetting the DataFrame index after arranging the readings chronologically
df = df.reset_index(drop=True)


# Counting duplicated timestamps for checking whether multiple observations exist at the same time
duplicate_timestamps = df["timestamp"].duplicated().sum()

# Displaying the number of duplicated timestamps found in the dataset
print("\nDuplicated timestamps:")
print(duplicate_timestamps)


# Counting missing values after the basic preprocessing steps
missing_values = df.isnull().sum()

# Displaying the number of missing values present in every column
print("\nMissing values:")
print(missing_values)


# Displaying the shape of the dataset after basic preprocessing
print("\nCleaned dataset shape:")
print(df.shape)


# Saving the cleaned and chronologically ordered dataset inside the processed data folder
df.to_csv(PROCESSED_DATA_PATH, index=False)

# Displaying the location of the saved cleaned dataset
print("\nCleaned dataset saved to:")
print(PROCESSED_DATA_PATH)