# Importing pandas for loading and checking the labelled MetroPT dataset
import pandas as pd

# Importing Path for creating and handling the labelled dataset file path
from pathlib import Path


# Finding the main PredictAgent project folder from the location of this Python file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Creating the complete path to the labelled MetroPT dataset
LABELLED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "metropt_labelled.csv"


# Loading only the columns required for checking the generated failure labels
df = pd.read_csv(
    LABELLED_DATA_PATH,
    usecols=["timestamp", "failure_active", "failure_next_6h"]
)

# Converting the timestamp column from text into datetime values for time-based checking
df["timestamp"] = pd.to_datetime(df["timestamp"])


# Selecting the readings labelled as having a failure starting within the next 6 hours
prediction_rows = df[df["failure_next_6h"] == 1]

# Selecting the readings occurring during an active documented failure
active_failure_rows = df[df["failure_active"] == 1]


# Displaying the earliest and latest timestamps receiving the next-6-hour failure label
print("Failure prediction label time range:")
print("First:", prediction_rows["timestamp"].min())
print("Last:", prediction_rows["timestamp"].max())


# Displaying the earliest and latest timestamps belonging to documented active failures
print("\nActive failure label time range:")
print("First:", active_failure_rows["timestamp"].min())
print("Last:", active_failure_rows["timestamp"].max())


# Identifying readings that were accidentally labelled as both an active failure and a future failure
overlap_count = (
    (df["failure_active"] == 1)
    & (df["failure_next_6h"] == 1)
).sum()

# Displaying the number of readings containing both labels at the same time
print("\nReadings labelled as both active failure and next-6-hour failure:")
print(overlap_count)