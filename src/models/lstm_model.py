# Importing PyTorch for creating the neural network
import torch

# Importing the neural network module for building the LSTM architecture
import torch.nn as nn


# Creating the LSTM model for predicting compressor failures
class LSTMFailurePredictor(nn.Module):

    # Initializing the LSTM architecture and its model parameters
    def __init__(
        self,
        input_size=15,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ):

        # Initializing the parent PyTorch neural network class
        super().__init__()


        # Creating the LSTM layers for learning temporal patterns from sensor sequences
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )


        # Creating a dropout layer for reducing overfitting
        self.dropout = nn.Dropout(dropout)


        # Creating the final output layer for producing one failure prediction score
        self.output_layer = nn.Linear(
            hidden_size,
            1
        )


    # Defining how sensor sequences move through the LSTM model
    def forward(self, x):

        # Passing the complete sensor sequence through the LSTM layers
        lstm_output, _ = self.lstm(x)


        # Selecting the LSTM representation from the final time step
        final_time_step = lstm_output[:, -1, :]


        # Applying dropout before generating the final prediction
        final_time_step = self.dropout(
            final_time_step
        )


        # Converting the learned representation into one raw prediction score
        prediction = self.output_layer(
            final_time_step
        )


        # Removing the unnecessary final dimension from the prediction tensor
        prediction = prediction.squeeze(1)


        # Returning the raw prediction score
        return prediction