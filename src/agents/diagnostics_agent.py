
# Importing NumPy for calculating sensor statistics
import numpy as np

# Importing pandas for processing sensor readings
import pandas as pd

# Importing Path for locating project files
from pathlib import Path


# Finding the main PredictAgent project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Creating the path to the cleaned MetroPT dataset
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "metropt_cleaned.csv"
)


# Selecting the continuous sensors used for diagnostics
SENSOR_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current"
]


# Selecting the digital operating-state signals
DIGITAL_COLUMNS = [
    "COMP",
    "DV_eletric",
    "MPG",
    "LPS",
    "Pressure_switch"
]


# Defining a class for compressor diagnostics
class DiagnosticsAgent:

    # Initializing the Diagnostics Agent
    def __init__(self):

        # Loading the cleaned sensor dataset
        self.data = pd.read_csv(
            DATA_PATH,
            parse_dates=["timestamp"]
        )

        # Sorting sensor readings chronologically
        self.data = self.data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # Creating reference statistics using the training period only
        training_data = self.data.loc[
            self.data["timestamp"] < "2020-06-01"
        ].copy()

        # Checking whether reference readings are available
        if training_data.empty:
            raise ValueError("No training reference data available.")

        # Creating separate reference statistics for operating states
        self.reference = {}

        # Assigning an operating state to each reference reading
        training_data["operating_state"] = training_data.apply(
            self.detect_operating_state,
            axis=1
        )

        # Calculating sensor statistics separately for each state
        for state, group in training_data.groupby("operating_state"):

            # Storing the number of reference readings for the state
            count = len(group)

            # Calculating mean sensor readings for the state
            means = group[SENSOR_COLUMNS].mean()

            # Calculating sensor standard deviations for the state
            stds = group[SENSOR_COLUMNS].std()

            # Saving the reference statistics
            self.reference[state] = {
                "count": count,
                "means": means,
                "stds": stds
            }

    # Detecting the compressor operating state from sensor signals
    @staticmethod
    def detect_operating_state(reading):

        # Selecting the measured motor current
        motor_current = float(reading["Motor_current"])

        # Selecting the electrical outlet-valve signal
        outlet_signal = int(reading["DV_eletric"])

        # Identifying readings with negligible motor current
        if motor_current < 1.0:
            return "INACTIVE"

        # Identifying readings with the outlet valve active
        if outlet_signal == 1:
            return "LOADED"

        # Identifying remaining running or offloaded readings
        return "OFFLOADED_OR_RUNNING"

    # Analyzing a window of recent sensor readings
    def analyze(self, recent_data):

        # Checking whether the input window contains readings
        if recent_data.empty:
            raise ValueError("The sensor window is empty.")

        # Checking whether all required sensor columns are available
        required_columns = SENSOR_COLUMNS + DIGITAL_COLUMNS

        missing_columns = [
            column
            for column in required_columns
            if column not in recent_data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing sensor columns: {missing_columns}"
            )

        # Selecting the latest sensor reading
        latest_reading = recent_data.iloc[-1]

        # Detecting the latest compressor operating state
        operating_state = self.detect_operating_state(
            latest_reading
        )

        # Selecting reference statistics for the detected state
        reference = self.reference.get(operating_state)

        # Checking whether a matching reference state exists
        if reference is None:
            raise ValueError(
                f"No reference statistics for {operating_state}"
            )

        # Creating a dictionary for storing sensor anomalies
        anomalies = {}

        # Comparing the latest sensor values with the state-specific baseline
        for sensor in SENSOR_COLUMNS:

            # Selecting the current sensor reading
            current_value = float(latest_reading[sensor])

            # Selecting the reference mean
            reference_mean = float(reference["means"][sensor])

            # Selecting the reference standard deviation
            reference_std = float(reference["stds"][sensor])

            # Skipping sensors without meaningful reference variation
            if not np.isfinite(reference_std) or reference_std <= 0:
                continue

            # Calculating the standardized difference from the baseline
            z_score = (
                current_value - reference_mean
            ) / reference_std

            # Flagging large deviations using an exploratory threshold
            if abs(z_score) >= 3.0:
                anomalies[sensor] = {
                    "current_value": round(current_value, 4),
                    "reference_mean": round(reference_mean, 4),
                    "z_score": round(float(z_score), 3)
                }

        # Identifying whether any unusual sensor readings were detected
        anomaly_detected = len(anomalies) > 0

        # Returning the diagnostic information
        return {
            "timestamp": str(latest_reading["timestamp"]),
            "operating_state": operating_state,
            "anomaly_detected": anomaly_detected,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "reference_readings": reference["count"],
            "diagnostic_method": "State-aware z-score screening",
            "failure_probability": None,
            "warning": (
                "Exploratory anomaly screening only; "
                "not a validated failure prediction."
            )
        }


# Running a local demonstration of the Diagnostics Agent
if __name__ == "__main__":

    # Creating the Diagnostics Agent
    agent = DiagnosticsAgent()

    # Selecting a sample window from the July pre-failure period
    sample_start = pd.Timestamp("2020-07-15 13:30:00")
    sample_end = pd.Timestamp("2020-07-15 14:29:59")

    # Extracting sensor readings from the selected period
    sample_data = agent.data.loc[
        (agent.data["timestamp"] >= sample_start)
        & (agent.data["timestamp"] <= sample_end)
    ].copy()

    # Running diagnostics on the selected sensor window
    result = agent.analyze(sample_data)

    # Displaying the diagnostic result
    print("\nDiagnostics Agent Result:")

    for key, value in result.items():
        print(f"{key}: {value}")
