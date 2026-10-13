# Importing pandas for loading and processing the datasets
import pandas as pd

# Importing Path for creating and handling project file paths
from pathlib import Path


# Finding the main PredictAgent project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating the path to the processed data folder
PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# Setting the number of sensor readings in each input window
WINDOW_SIZE = 360


# Moving approximately one minute forward between consecutive windows
STRIDE = 6


# Setting the expected duration of each sensor window in minutes
EXPECTED_DURATION_MINUTES = 60


# Allowing a five-minute tolerance because timestamps are not perfectly regular
DURATION_TOLERANCE_MINUTES = 5


# Creating a function for generating valid sequence windows
def create_windows(data, dataset_name):

    # Converting timestamps into datetime values
    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )


    # Resetting the row indices for reliable sequence slicing
    data = data.reset_index(drop=True)


    # Creating an empty list for storing valid window information
    windows = []


    # Starting the count of windows rejected because they contain active failures
    rejected_failure_windows = 0


    # Starting the count of windows rejected because their duration is unsuitable
    rejected_duration_windows = 0


    # Moving through the dataset using the selected stride
    for start_index in range(
        0,
        len(data) - WINDOW_SIZE + 1,
        STRIDE
    ):

        # Calculating the final row index of the current window
        end_index = (
            start_index
            + WINDOW_SIZE
            - 1
        )


        # Selecting the complete 360-reading window
        window_data = data.iloc[
            start_index:end_index + 1
        ]


        # Rejecting any window containing even one active-failure reading
        if window_data["failure_active"].max() == 1:

            rejected_failure_windows += 1
            continue


        # Selecting the timestamp at the beginning of the window
        start_time = window_data["timestamp"].iloc[0]


        # Selecting the timestamp at the end of the window
        end_time = window_data["timestamp"].iloc[-1]


        # Calculating the actual duration of the window in minutes
        duration_minutes = (
            end_time - start_time
        ).total_seconds() / 60


        # Calculating the minimum acceptable duration
        minimum_duration = (
            EXPECTED_DURATION_MINUTES
            - DURATION_TOLERANCE_MINUTES
        )


        # Calculating the maximum acceptable duration
        maximum_duration = (
            EXPECTED_DURATION_MINUTES
            + DURATION_TOLERANCE_MINUTES
        )


        # Rejecting windows with unsuitable timestamp durations
        if not (
            minimum_duration
            <= duration_minutes
            <= maximum_duration
        ):

            rejected_duration_windows += 1
            continue


        # Selecting the prediction target at the end of the input window
        target = int(
            window_data["failure_next_6h"].iloc[-1]
        )


        # Storing the valid window information
        windows.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": duration_minutes,
                "target": target
            }
        )


    # Converting the valid window information into a DataFrame
    windows_df = pd.DataFrame(
        windows
    )


    # Displaying the dataset being processed
    print(f"\n{dataset_name}")


    # Displaying the number of valid windows
    print(
        "Valid windows:",
        len(windows_df)
    )


    # Displaying the number of windows rejected for containing active failures
    print(
        "Rejected active-failure windows:",
        rejected_failure_windows
    )


    # Displaying the number of windows rejected because of duration
    print(
        "Rejected duration windows:",
        rejected_duration_windows
    )


    # Displaying the target distribution
    print("\nTarget distribution:")

    print(
        windows_df["target"].value_counts()
    )


    # Displaying the minimum valid window duration
    print(
        "\nMinimum duration:",
        windows_df["duration_minutes"].min()
    )


    # Displaying the maximum valid window duration
    print(
        "Maximum duration:",
        windows_df["duration_minutes"].max()
    )


    # Returning all valid windows
    return windows_df


# Creating information for each chronological dataset split
datasets = {
    "Training": (
        "train_scaled.csv",
        "train_windows.csv"
    ),
    "Validation": (
        "validation_scaled.csv",
        "validation_windows.csv"
    ),
    "Test": (
        "test_scaled.csv",
        "test_windows.csv"
    )
}


# Processing each chronological dataset split
for dataset_name, filenames in datasets.items():

    # Separating the input and output filenames
    data_filename, windows_filename = filenames


    # Creating the path to the scaled sensor dataset
    data_path = (
        PROCESSED_DATA_PATH
        / data_filename
    )


    # Creating the path where the new window metadata will be saved
    windows_path = (
        PROCESSED_DATA_PATH
        / windows_filename
    )


    # Loading the scaled sensor dataset
    data = pd.read_csv(
        data_path
    )


    # Creating clean failure-free input windows
    windows = create_windows(
        data,
        dataset_name
    )


    # Saving the corrected window metadata
    windows.to_csv(
        windows_path,
        index=False
    )


    # Displaying the output location
    print(
        "\nSaved to:",
        windows_path
    )