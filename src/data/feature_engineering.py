# Importing pandas for loading the scaled datasets and window information
import pandas as pd

# Importing numpy for calculating numerical statistical features
import numpy as np

# Importing Path for creating and handling project file paths
from pathlib import Path


# Finding the main PredictAgent project folder from the location of this Python file
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating paths to the scaled train, validation, and test datasets
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_scaled.csv"
VALIDATION_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "validation_scaled.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_scaled.csv"


# Creating paths to the previously generated 60-minute window information
TRAIN_WINDOWS_PATH = PROJECT_ROOT / "data" / "processed" / "train_windows.csv"
VALIDATION_WINDOWS_PATH = PROJECT_ROOT / "data" / "processed" / "validation_windows.csv"
TEST_WINDOWS_PATH = PROJECT_ROOT / "data" / "processed" / "test_windows.csv"


# Creating paths where the engineered feature datasets will be saved
TRAIN_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
VALIDATION_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "validation_features.csv"
TEST_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"


# Listing the 7 continuous analogue sensor measurements
analogue_columns = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current"
]


# Listing the 8 digital compressor signals
digital_columns = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses"
]


# Creating a function for converting 60-minute windows into summary features
def create_features(data_path, windows_path, output_path, dataset_name):

    # Loading the scaled sensor dataset
    df = pd.read_csv(data_path)

    # Loading the start and end positions of the previously generated windows
    windows_df = pd.read_csv(windows_path)

    # Creating an empty list for storing the features generated from every window
    feature_rows = []


    # Looping through every valid 60-minute window
    for _, window_info in windows_df.iterrows():

        # Getting the starting row position of the current window
        start_index = int(window_info["start_index"])

        # Getting the ending row position of the current window
        end_index = int(window_info["end_index"])

        # Extracting all sensor readings belonging to the current 60-minute window
        window_data = df.iloc[start_index:end_index + 1]

        # Creating a dictionary for storing the features of the current window
        features = {}


        # Looping through every continuous analogue sensor
        for column in analogue_columns:

            # Calculating the average sensor value during the 60-minute window
            features[f"{column}_mean"] = window_data[column].mean()

            # Calculating the standard deviation for measuring variation in the sensor values
            features[f"{column}_std"] = window_data[column].std()

            # Calculating the minimum sensor value observed during the window
            features[f"{column}_min"] = window_data[column].min()

            # Calculating the maximum sensor value observed during the window
            features[f"{column}_max"] = window_data[column].max()

            # Calculating the difference between the maximum and minimum sensor values
            features[f"{column}_range"] = (
                window_data[column].max() - window_data[column].min()
            )


        # Looping through every digital compressor signal
        for column in digital_columns:

            # Calculating the proportion of the window during which the digital signal was active
            features[f"{column}_mean"] = window_data[column].mean()

            # Counting how many times the digital signal changed its state during the window
            features[f"{column}_changes"] = (
                window_data[column].diff().fillna(0) != 0
            ).sum()


        # Storing the ending timestamp for identifying when the prediction would have been made
        features["window_end_time"] = window_info["end_time"]

        # Storing the prediction target associated with the current window
        features["target"] = int(window_info["target"])

        # Adding the completed feature dictionary to the collection of feature rows
        feature_rows.append(features)


    # Converting all generated feature dictionaries into a pandas DataFrame
    features_df = pd.DataFrame(feature_rows)

    # Saving the engineered features for later model training and evaluation
    features_df.to_csv(output_path, index=False)


    # Displaying the number of windows converted into feature rows
    print(f"\n{dataset_name} feature dataset shape:")
    print(features_df.shape)

    # Displaying the number of positive prediction examples in the generated feature dataset
    print("Positive examples:")
    print(features_df["target"].sum())

    # Displaying the number of negative prediction examples in the generated feature dataset
    print("Negative examples:")
    print((features_df["target"] == 0).sum())

    # Displaying the total number of model input features created
    print("Number of engineered input features:")
    print(len(features_df.columns) - 2)


# Creating engineered features for the training windows
create_features(
    TRAIN_DATA_PATH,
    TRAIN_WINDOWS_PATH,
    TRAIN_FEATURES_PATH,
    "Training"
)


# Creating engineered features for the validation windows
create_features(
    VALIDATION_DATA_PATH,
    VALIDATION_WINDOWS_PATH,
    VALIDATION_FEATURES_PATH,
    "Validation"
)


# Creating engineered features for the test windows
create_features(
    TEST_DATA_PATH,
    TEST_WINDOWS_PATH,
    TEST_FEATURES_PATH,
    "Test"
)