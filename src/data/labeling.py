# Importing pandas for loading the sensor dataset and failure information
import pandas as pd

# Importing Path for creating paths to the project data files
from pathlib import Path


# Finding the main PredictAgent project folder from the location of this Python file
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Creating the path to the cleaned MetroPT sensor dataset
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "metropt_cleaned.csv"

# Creating the path to the documented MetroPT failure information
FAILURE_DATA_PATH = PROJECT_ROOT / "data" / "failure_metadata" / "failures.csv"

# Creating the path where the labelled dataset will be saved
LABELLED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "metropt_labelled.csv"


# Loading the cleaned MetroPT sensor dataset
df = pd.read_csv(CLEANED_DATA_PATH)

# Loading the documented compressor failure periods
failures = pd.read_csv(FAILURE_DATA_PATH)


# Converting the sensor timestamps from text into datetime values
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Converting the failure start times from text into datetime values
failures["start_time"] = pd.to_datetime(failures["start_time"])

# Converting the failure end times from text into datetime values
failures["end_time"] = pd.to_datetime(failures["end_time"])


# Creating a column initially marking every sensor reading as outside a failure period
df["failure_active"] = 0

# Creating a column initially marking every sensor reading as having no failure starting within the next 6 hours
df["failure_next_6h"] = 0


# Looping through every documented compressor failure
for _, failure in failures.iterrows():

    # Storing the starting timestamp of the current documented failure
    failure_start = failure["start_time"]

    # Storing the ending timestamp of the current documented failure
    failure_end = failure["end_time"]

    # Calculating the beginning of the 6-hour early-warning period before the failure starts
    warning_start = failure_start - pd.Timedelta(hours=6)


    # Identifying sensor readings occurring during the documented failure period
    active_mask = (
        (df["timestamp"] >= failure_start)
        & (df["timestamp"] <= failure_end)
    )

    # Marking readings occurring during the documented failure period as active failures
    df.loc[active_mask, "failure_active"] = 1


    # Identifying healthy pre-failure readings occurring during the 6 hours before the failure begins
    prediction_mask = (
        (df["timestamp"] >= warning_start)
        & (df["timestamp"] < failure_start)
        & (df["failure_active"] == 0)
    )

    # Marking these pre-failure readings as positive examples for 6-hour-ahead failure prediction
    df.loc[prediction_mask, "failure_next_6h"] = 1


# Counting the number of readings occurring during documented failure periods
active_failure_count = df["failure_active"].sum()

# Counting the number of readings belonging to the 6-hour pre-failure prediction periods
prediction_count = df["failure_next_6h"].sum()


# Displaying the number of readings marked as active failures
print("Readings during documented failures:")
print(active_failure_count)

# Displaying the number of readings marked as failure-within-6-hours examples
print("\nReadings labelled as failure within the next 6 hours:")
print(prediction_count)

# Displaying the number of readings that remain normal prediction examples
print("\nReadings labelled as no failure within the next 6 hours:")
print((df["failure_next_6h"] == 0).sum())


# Saving the sensor readings together with their newly created failure labels
df.to_csv(LABELLED_DATA_PATH, index=False)

# Displaying the location of the newly created labelled dataset
print("\nLabelled dataset saved to:")
print(LABELLED_DATA_PATH)