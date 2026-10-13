# Importing pandas for loading the scaled sensor data and window information
import pandas as pd

# Importing numpy for converting sensor windows into numerical arrays
import numpy as np

# Importing torch for creating tensors used by the LSTM model
import torch

# Importing Dataset for creating a custom PyTorch sequence dataset
from torch.utils.data import Dataset


# Creating a custom dataset for loading 60-minute compressor sensor sequences
class MetroPTSequenceDataset(Dataset):

    # Initializing the dataset using the scaled sensor file and window metadata file
    def __init__(self, data_path, windows_path):

        # Loading the scaled sensor dataset
        self.data = pd.read_csv(data_path)

        # Loading the previously generated window information
        self.windows = pd.read_csv(windows_path)


        # Listing the 15 sensor and compressor signals used as LSTM inputs
        self.feature_columns = [
            "TP2",
            "TP3",
            "H1",
            "DV_pressure",
            "Reservoirs",
            "Oil_temperature",
            "Motor_current",
            "COMP",
            "DV_eletric",
            "Towers",
            "MPG",
            "LPS",
            "Pressure_switch",
            "Oil_level",
            "Caudal_impulses"
        ]


        # Converting the sensor features into a float32 NumPy array for faster sequence extraction
        self.sensor_values = self.data[
            self.feature_columns
        ].to_numpy(dtype=np.float32)


        # Converting the window starting positions into an integer NumPy array
        self.start_indices = self.windows[
            "start_index"
        ].to_numpy(dtype=np.int64)


        # Converting the window ending positions into an integer NumPy array
        self.end_indices = self.windows[
            "end_index"
        ].to_numpy(dtype=np.int64)


        # Converting the failure prediction targets into a float32 NumPy array
        self.targets = self.windows[
            "target"
        ].to_numpy(dtype=np.float32)


    # Returning the total number of available prediction windows
    def __len__(self):

        return len(self.windows)


    # Loading one 60-minute sequence and its corresponding prediction target
    def __getitem__(self, index):

        # Getting the starting row position of the requested sequence
        start_index = self.start_indices[index]

        # Getting the ending row position of the requested sequence
        end_index = self.end_indices[index]


        # Extracting the 360 time steps and 15 sensor signals belonging to the sequence
        sequence = self.sensor_values[
            start_index:end_index + 1
        ]


        # Getting the failure-within-6-hours target for the sequence
        target = self.targets[index]


        # Converting the sensor sequence into a PyTorch tensor
        sequence_tensor = torch.tensor(
            sequence,
            dtype=torch.float32
        )


        # Converting the prediction target into a PyTorch tensor
        target_tensor = torch.tensor(
            target,
            dtype=torch.float32
        )


        # Returning the sequence and its corresponding target
        return sequence_tensor, target_tensor