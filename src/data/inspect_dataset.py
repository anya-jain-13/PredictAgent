# Importing pandas for loading and analysing the MetroPT dataset
import pandas as pd

# Importing Path for creating and handling the dataset file path
from pathlib import Path


# Finding the main PredictAgent project folder from the location of this Python file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Creating the complete path to the MetroPT Air Compressor CSV dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"


# Printing a heading for clearly separating the dataset inspection output
print("=" * 60)
print("PREDICTAGENT - METROPT DATASET INSPECTION")
print("=" * 60)


# Printing the dataset path for confirming that the correct CSV file is being accessed
print("\nLoading dataset from:")
print(DATA_PATH)


# Loading the MetroPT CSV file into a pandas DataFrame
df = pd.read_csv(DATA_PATH)


# Displaying the number of rows and columns present in the MetroPT dataset
print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)


# Displaying every column name for identifying the available sensor and system variables
print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

# Looping through all columns for displaying each column number together with its name
for i, column in enumerate(df.columns):
    print(f"{i}: {column}")

#Python repeats "=" 60 times
# Displaying the first five rows for understanding how the MetroPT observations are structured
print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())


# Displaying the data type of every column for understanding how each variable is stored
print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)


# Counting missing values in every column for identifying incomplete sensor or system readings
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

# Converting the timestamp column from text into datetime values for performing time-based calculations
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Calculating the time difference between each sensor reading and the previous sensor reading
time_differences = df["timestamp"].diff()

# Counting how frequently each time interval occurs for identifying the actual sampling interval of the dataset
print("\nTIME INTERVALS BETWEEN READINGS")
print(time_differences.value_counts().head(10))

#CSV is effectively sampled about every 10 seconds, not every second.
#1 minute  ≈ 6 readings
#1 hour    ≈ 360 readings
#6 hours   ≈ 2,160 readings