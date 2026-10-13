
# Importing pandas for loading and analyzing sensor readings
import pandas as pd

# Importing NumPy for calculating numerical statistics
import numpy as np

# Importing Matplotlib for visualizing sensor behaviour
import matplotlib.pyplot as plt

# Importing Path for handling project file locations
from pathlib import Path


# Finding the main PredictAgent project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating the path to the cleaned sensor dataset
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "metropt_cleaned.csv"
)


# Creating the path to the documented failure information
FAILURE_PATH = (
    PROJECT_ROOT / "data" / "failure_metadata" / "failures.csv"
)


# Creating the folder for saving analysis results
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# Selecting the seven continuous sensor signals
SENSOR_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current"
]


# Loading the cleaned MetroPT sensor dataset
data = pd.read_csv(
    DATA_PATH,
    parse_dates=["timestamp"]
)


# Loading the documented air-leak failure periods
failures = pd.read_csv(
    FAILURE_PATH,
    parse_dates=["start_time", "end_time"]
)


# Sorting sensor readings chronologically
data = data.sort_values("timestamp").reset_index(drop=True)


# Creating an empty list for storing event-level statistics
event_results = []


# Creating an empty list for storing sensor comparison statistics
comparison_results = []


# Creating a mask for identifying readings near documented failures
excluded_from_baseline = pd.Series(
    False,
    index=data.index
)


# Excluding failure periods and the preceding six hours from the baseline
for _, failure in failures.iterrows():

    start_time = failure["start_time"]
    end_time = failure["end_time"]

    excluded_from_baseline |= (
        (data["timestamp"] >= start_time - pd.Timedelta(hours=6))
        & (data["timestamp"] <= end_time)
    )


# Selecting readings outside all documented failure-related periods
normal_data = data.loc[
    ~excluded_from_baseline,
    SENSOR_COLUMNS
]


# Calculating the baseline mean for each sensor
baseline_means = normal_data.mean()


# Calculating the baseline standard deviation for each sensor
baseline_stds = normal_data.std()


# Displaying the number of baseline sensor readings
print("\nNormal baseline readings:")
print(len(normal_data))


# Analyzing each documented failure separately
for _, failure in failures.iterrows():

    # Selecting the failure identifier
    failure_id = int(failure["failure_id"])

    # Selecting the documented failure start time
    failure_start = failure["start_time"]

    # Calculating the beginning of the six-hour prediction period
    prediction_start = failure_start - pd.Timedelta(hours=6)


    # Selecting sensor readings before the failure begins
    pre_failure_data = data.loc[
        (data["timestamp"] >= prediction_start)
        & (data["timestamp"] < failure_start)
    ].copy()


    # Displaying the current failure being analyzed
    print(f"\nFailure {failure_id}")
    print("Failure start:", failure_start)
    print("Pre-failure readings:", len(pre_failure_data))


    # Skipping the event when no preceding readings are available
    if pre_failure_data.empty:
        print("No pre-failure sensor readings available.")
        continue


    # Storing summary information for the failure event
    event_results.append({
        "failure_id": failure_id,
        "failure_start": failure_start,
        "pre_failure_readings": len(pre_failure_data),
        "first_reading": pre_failure_data["timestamp"].min(),
        "last_reading": pre_failure_data["timestamp"].max()
    })


    # Comparing each sensor with the normal operating baseline
    for sensor in SENSOR_COLUMNS:

        # Calculating the mean sensor value before the failure
        event_mean = pre_failure_data[sensor].mean()

        # Calculating the normal operating mean
        baseline_mean = baseline_means[sensor]

        # Calculating the normal operating standard deviation
        baseline_std = baseline_stds[sensor]


        # Calculating the standardized difference from normal behaviour
        if baseline_std > 0:
            standardized_difference = (
                event_mean - baseline_mean
            ) / baseline_std
        else:
            standardized_difference = np.nan


        # Storing the sensor comparison results
        comparison_results.append({
            "failure_id": failure_id,
            "sensor": sensor,
            "pre_failure_mean": event_mean,
            "normal_mean": baseline_mean,
            "normal_std": baseline_std,
            "standardized_difference": standardized_difference
        })


# Converting the event summaries into a DataFrame
events_df = pd.DataFrame(event_results)


# Converting the sensor comparisons into a DataFrame
comparison_df = pd.DataFrame(comparison_results)


# Saving the event-level summary
events_df.to_csv(
    OUTPUT_PATH / "failure_event_summary.csv",
    index=False
)


# Saving the sensor comparison results
comparison_df.to_csv(
    OUTPUT_PATH / "failure_sensor_comparison.csv",
    index=False
)


# Creating a table comparing standardized sensor differences across failures
comparison_table = comparison_df.pivot(
    index="sensor",
    columns="failure_id",
    values="standardized_difference"
)


# Displaying the sensor comparison table
print("\nStandardized sensor differences from normal operation:")
print(comparison_table.round(3).to_string())


# Creating a heatmap of sensor differences across failure events
fig, ax = plt.subplots(figsize=(10, 6))


# Displaying standardized differences using a diverging colour scale
heatmap = ax.imshow(
    comparison_table,
    cmap="coolwarm",
    aspect="auto",
    vmin=-3,
    vmax=3
)


# Displaying the sensor names on the vertical axis
ax.set_yticks(range(len(comparison_table.index)))
ax.set_yticklabels(comparison_table.index)


# Displaying the failure identifiers on the horizontal axis
ax.set_xticks(range(len(comparison_table.columns)))
ax.set_xticklabels(
    [f"Failure {i}" for i in comparison_table.columns]
)


# Adding the standardized difference inside each heatmap cell
for row in range(comparison_table.shape[0]):
    for col in range(comparison_table.shape[1]):

        value = comparison_table.iloc[row, col]

        ax.text(
            col,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=9
        )


# Adding a descriptive chart title
ax.set_title(
    "Sensor Behaviour During the 6 Hours Before Each Failure"
)


# Adding a colour scale to explain the standardized differences
fig.colorbar(
    heatmap,
    ax=ax,
    label="Difference from normal mean (standard deviations)"
)


# Adjusting chart spacing
plt.tight_layout()


# Saving the failure comparison heatmap
plt.savefig(
    OUTPUT_PATH / "failure_event_heatmap.png",
    dpi=200
)


# Displaying the completed heatmap
plt.show()


# Displaying confirmation that the analysis finished
print("\nFailure-event analysis completed.")
print("Saved failure_event_summary.csv")
print("Saved failure_sensor_comparison.csv")
print("Saved failure_event_heatmap.png")


# Selecting important sensors for analyzing pre-failure trends
TREND_SENSORS = [
    "Oil_temperature",
    "Motor_current",
    "TP2"
]


# Creating trend graphs for each documented failure
for _, failure in failures.iterrows():

    # Selecting the failure identifier and starting time
    failure_id = int(failure["failure_id"])
    failure_start = failure["start_time"]


    # Selecting the six-hour period before failure
    pre_failure_data = data.loc[
        (data["timestamp"] >= failure_start - pd.Timedelta(hours=6))
        & (data["timestamp"] < failure_start)
    ].copy()


    # Skipping failures without available preceding readings
    if pre_failure_data.empty:
        continue


    # Calculating the time remaining until failure in hours
    pre_failure_data["hours_before_failure"] = (
        pre_failure_data["timestamp"] - failure_start
    ).dt.total_seconds() / 3600


    # Grouping sensor readings into ten-minute intervals to reduce noise
    pre_failure_data = (
        pre_failure_data
        .set_index("timestamp")
        .resample("10min")[TREND_SENSORS]
        .mean()
        .dropna()
        .reset_index()
    )


    # Recalculating time remaining until failure after resampling
    pre_failure_data["hours_before_failure"] = (
        pre_failure_data["timestamp"] - failure_start
    ).dt.total_seconds() / 3600


    # Creating three vertically arranged sensor graphs
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True
    )


    # Plotting each sensor's readings before failure
    for index, sensor in enumerate(TREND_SENSORS):

        axes[index].plot(
            pre_failure_data["hours_before_failure"],
            pre_failure_data[sensor],
            linewidth=2
        )

        axes[index].set_ylabel(sensor)
        axes[index].grid(True, alpha=0.3)


    # Adding the failure event title
    fig.suptitle(
        f"Sensor Trends Before Failure {failure_id}",
        fontsize=15
    )


    # Labelling the time remaining until failure
    axes[-1].set_xlabel(
        "Hours Relative to Failure Start"
    )


    # Adjusting the chart layout
    plt.tight_layout()


    # Saving the trend graph for the current failure
    plt.savefig(
        OUTPUT_PATH / f"failure_{failure_id}_trends.png",
        dpi=200
    )


    # Displaying the trend graph
    plt.show()


# Displaying confirmation that all trend graphs were created
print("\nAll pre-failure trend graphs completed.")

# Selecting signals for investigating compressor operating states
STATE_COLUMNS = [
    "Motor_current",
    "COMP",
    "DV_eletric",
    "MPG",
    "LPS",
    "Pressure_switch"
]


# Comparing compressor operating signals before each failure
for _, failure in failures.iterrows():

    # Selecting the failure identifier and start time
    failure_id = int(failure["failure_id"])
    failure_start = failure["start_time"]

    # Selecting the six hours before the failure
    pre_failure_data = data.loc[
        (data["timestamp"] >= failure_start - pd.Timedelta(hours=6))
        & (data["timestamp"] < failure_start)
    ].copy()

    # Displaying the failure being investigated
    print(f"\nFailure {failure_id} - Operating State Analysis")

    # Skipping events without sensor readings
    if pre_failure_data.empty:
        print("No readings available.")
        continue

    # Displaying the average motor current
    print(
        "Average motor current:",
        round(pre_failure_data["Motor_current"].mean(), 3)
    )

    # Displaying the percentage of readings with very low motor current
    print(
        "Motor current below 1 A (%):",
        round(
            (pre_failure_data["Motor_current"] < 1).mean() * 100,
            2
        )
    )

    # Examining the observed values of the digital operating signals
    for signal in STATE_COLUMNS[1:]:

        # Counting the occurrences of each signal value
        counts = pre_failure_data[signal].value_counts(
            normalize=True
        ) * 100

        # Displaying the signal distribution
        print(f"\n{signal} distribution (%):")
        print(counts.round(2).to_string())

